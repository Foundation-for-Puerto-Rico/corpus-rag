#!/usr/bin/env python3
"""Config-driven RAG Retriever — Semantic, FTS, and Hybrid search over
documents stored in LanceDB with Nomic Embed v1.5 embeddings.

All domain-specific logic (valid filters, result fields, document precedence)
comes from a CorpusConfig. Falls back to sensible defaults when no config
is provided.

Uses ONNX Runtime for inference (no PyTorch dependency).

Usage:
    from retriever import Retriever
    from config_loader import load_config

    config = load_config("corpus_config.yaml")
    r = Retriever(config=config)
    results = r.hybrid_search("search query", limit=5)
"""

import logging
import os
import re as _re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FTS query escaping
# ---------------------------------------------------------------------------

_FTS_SPECIAL = _re.compile(r'([+\-!(){}[\]^"~*?:\\/])')


def _escape_fts_query(query: str) -> str:
    """Escape special characters for Tantivy FTS queries."""
    return _FTS_SPECIAL.sub(r"\\\1", query)


# ---------------------------------------------------------------------------
# Config-driven builder functions
# ---------------------------------------------------------------------------

# Filters always available regardless of config
_ALWAYS_PRESENT_FILTERS = {
    "content_type",
    "has_quantitative_data",
    "document_type",
    "is_summary",
}

# Extra fields the retriever always includes beyond config.result_fields()
_RETRIEVER_EXTRA_FIELDS = ["context", "page_start", "page_end"]

# Default field values for backward compatibility with older DBs
_BASE_FIELD_DEFAULTS = {
    "document_type": "unknown",
    "page_start": -1,
    "page_end": -1,
    "context": "",
    "is_summary": False,
}


def build_valid_filters(config) -> set[str]:
    """Build set of valid filter names from config."""
    return config.valid_filters()


def build_result_fields(config) -> list[str]:
    """Build ordered list of result fields from config.

    Includes config-defined fields plus retriever-specific extras
    (context, page_start, page_end).
    """
    fields = config.result_fields()
    for extra in _RETRIEVER_EXTRA_FIELDS:
        if extra not in fields:
            fields.append(extra)
    return fields


def build_precedence_boost(config) -> dict[str, float]:
    """Build document-type to precedence boost mapping from config."""
    return config.precedence_boost_map()


def build_field_defaults(config) -> dict[str, object]:
    """Build field defaults dict combining base defaults with config entities/custom fields."""
    defaults = dict(_BASE_FIELD_DEFAULTS)
    for e in config.entities:
        if e.name not in defaults:
            defaults[e.name] = ""
    for cf in config.custom_fields:
        if cf.name not in defaults:
            if cf.type == "bool":
                defaults[cf.name] = False
            elif cf.type == "int":
                defaults[cf.name] = 0
            else:
                defaults[cf.name] = ""
    return defaults


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Config-driven retriever for semantic, FTS, and hybrid search."""

    SUMMARY_DEMOTION = 0.70  # Summaries are useful but shouldn't displace specific content

    def __init__(
        self,
        config=None,
        db_path: str = "",
        table_name: str = "",
    ):
        from config_loader import CorpusConfig

        self._config: CorpusConfig | None = config

        # Build config-driven lookups
        if config is not None:
            self._valid_filters = build_valid_filters(config)
            self._result_fields = build_result_fields(config)
            self._precedence_boost = build_precedence_boost(config)
            self._field_defaults = build_field_defaults(config)
        else:
            self._valid_filters = set(_ALWAYS_PRESENT_FILTERS)
            self._result_fields = [
                "text", "context", "section_title", "content_type",
                "has_quantitative_data", "document_type",
                "page_start", "page_end", "is_summary",
            ]
            self._precedence_boost = {}
            self._field_defaults = dict(_BASE_FIELD_DEFAULTS)

        # Resolve paths
        _db_path = db_path or os.environ.get(
            "CORPUS_DB_PATH", str(Path.home() / "corpus-rag" / "data")
        )
        _table_name = table_name or (
            config.project.table_name if config else "chunks"
        )

        import lancedb

        self._db = lancedb.connect(_db_path)
        self._table = self._db.open_table(_table_name)

        # Lazy-loaded ONNX model
        self._session = None
        self._tokenizer = None

    # -- Filter builder ----------------------------------------------------

    def _build_where(self, **filters) -> str:
        """Build a SQL WHERE clause from keyword filters.

        Supports string, bool, and None values. Unknown filter keys are
        silently ignored. Single quotes in string values are escaped.
        """
        clauses = []
        for key, val in filters.items():
            if key not in self._valid_filters or val is None:
                continue
            if isinstance(val, bool):
                clauses.append(f"{key} = {'true' if val else 'false'}")
            elif isinstance(val, str):
                escaped = val.replace("'", "''")
                clauses.append(f"{key} = '{escaped}'")
        return " AND ".join(clauses)

    # -- Model loading (lazy) ---------------------------------------------

    def _load_model(self):
        """Lazy-load ONNX model and tokenizer on first use."""
        if self._session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_dir = os.environ.get(
                "CORPUS_ONNX_MODEL_DIR",
                str(Path.home() / "corpus-rag" / "models" / "nomic-embed-text-v1.5"),
            )
            model_path = os.path.join(model_dir, "model.onnx")
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(model_path, sess_options=opts)
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=2048)
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def _encode_query(self, query: str):
        """Encode a query string into a normalized embedding vector."""
        import numpy as np

        self._load_model()
        text = f"search_query: {query}"
        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # outputs[0] is last_hidden_state: (1, seq_len, 768)
        token_embeddings = outputs[0]

        # Mean pooling (respecting attention mask)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        count = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / count

        # L2 normalize
        norm = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        normalized = mean_pooled / np.clip(norm, a_min=1e-9, a_max=None)

        return normalized[0]

    # -- Precedence boost --------------------------------------------------

    def _apply_precedence_boost(self, results: list[dict]) -> list[dict]:
        """Apply document-type precedence boost and re-rank results."""
        for r in results:
            doc_type = r.get("document_type", "unknown")
            boost = self._precedence_boost.get(doc_type, 1.0)
            if r.get("is_summary"):
                boost *= self.SUMMARY_DEMOTION
            r["original_score"] = r["score"]
            r["score"] = round(r["score"] * boost, 6)
            r["precedence_boost"] = boost
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    # -- Semantic search ---------------------------------------------------

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        apply_precedence: bool = True,
        **filters,
    ) -> list[dict]:
        """Search using vector similarity (cosine distance)."""
        if not query or not query.strip():
            return []
        embedding = self._encode_query(query)
        q = self._table.search(embedding).limit(limit)
        where = self._build_where(**filters)
        if where:
            q = q.where(where)
        df = q.to_pandas()
        results = self._format_results(df, "_distance", higher_is_better=False)
        if apply_precedence:
            results = self._apply_precedence_boost(results)
        return results

    # -- FTS search --------------------------------------------------------

    def fts_search(
        self,
        query: str,
        limit: int = 5,
        apply_precedence: bool = True,
        **filters,
    ) -> list[dict]:
        """Search using full-text search (Tantivy)."""
        if not query or not query.strip():
            return []
        escaped = _escape_fts_query(query)
        q = self._table.search(escaped, query_type="fts").limit(limit)
        where = self._build_where(**filters)
        if where:
            q = q.where(where)
        df = q.to_pandas()
        results = self._format_results(df, "_score", higher_is_better=True)
        if apply_precedence:
            results = self._apply_precedence_boost(results)
        return results

    # -- Hybrid search (RRF fusion) ----------------------------------------

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        semantic_weight: float = 0.5,
        fts_weight: float | None = None,
        rrf_k: int = 60,
        apply_precedence: bool = True,
        **filters,
    ) -> list[dict]:
        """Search using Reciprocal Rank Fusion of semantic + FTS results."""
        if not query or not query.strip():
            return []

        if fts_weight is None:
            fts_weight = 1.0 - semantic_weight

        fetch = limit * 3  # over-fetch for better fusion

        # Run both searches (no boost here -- applied after RRF fusion)
        sem_results = self.semantic_search(
            query, limit=fetch, apply_precedence=False, **filters
        )
        fts_results = self.fts_search(
            query, limit=fetch, apply_precedence=False, **filters
        )

        # Build RRF scores keyed by (first entity or doc_type, chunk text hash)
        def _key(d):
            # Use first entity name as key component if available, else document_type
            entity_names = (
                [e.name for e in self._config.entities] if self._config else []
            )
            key_val = d.get(entity_names[0], "") if entity_names else d.get("document_type", "")
            return (key_val, hash(d["text"][:200]))

        rrf_scores: dict[tuple, float] = {}
        doc_data: dict[tuple, dict] = {}

        for d in sem_results:
            k = _key(d)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + semantic_weight / (
                rrf_k + d["rank"]
            )
            doc_data[k] = d

        for d in fts_results:
            k = _key(d)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + fts_weight / (
                rrf_k + d["rank"]
            )
            if k not in doc_data:
                doc_data[k] = d

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:limit]

        # Normalize scores
        if sorted_keys:
            max_score = rrf_scores[sorted_keys[0]]
            results = []
            for i, k in enumerate(sorted_keys):
                d = dict(doc_data[k])
                d["score"] = rrf_scores[k] / max_score if max_score > 0 else 0.0
                d["rank"] = i + 1
                results.append(d)
            if apply_precedence:
                results = self._apply_precedence_boost(results)
            return results
        return []

    # -- Result formatter --------------------------------------------------

    def _format_results(
        self, df, score_col: str, higher_is_better: bool
    ) -> list[dict]:
        """Normalize scores and return a list of result dicts.

        Parameters
        ----------
        df : pandas.DataFrame
            Search results from LanceDB.
        score_col : str
            Column name containing the raw score.
        higher_is_better : bool
            If True, scores are normalized by dividing by max.
            If False, scores are treated as distances (lower = better)
            and converted to similarity via ``1 - distance``.
        """
        if df.empty:
            return []
        scores = df[score_col].values
        if higher_is_better:
            norm = scores / scores.max() if scores.max() > 0 else scores
        else:
            norm = 1.0 - scores  # cosine distance -> similarity
        df_cols = set(df.columns)
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            d = {}
            for f in self._result_fields:
                if f in df_cols:
                    d[f] = row[f]
                else:
                    d[f] = self._field_defaults.get(f, "")
            if d.get("is_summary"):
                d["text"] = "[RESUMEN IA] " + d.get("text", "")
            d["score"] = float(norm[i])
            d["rank"] = i + 1
            results.append(d)
        return results

    # -- Re-ranking --------------------------------------------------------

    _RERANK_PROMPT = """Evalúa la relevancia de cada resultado para esta consulta de búsqueda.
Asigna un score de 0 a 10 (10 = perfectamente relevante).

Consulta: {query}

{results_block}

Responde SOLO en JSON: [{{"index": 0, "score": N}}, ...]"""

    def _build_rerank_block(self, results: list[dict]) -> str:
        """Build text block for LLM re-ranking with dynamic entity info."""
        entity_names = (
            [e.name for e in self._config.entities] if self._config else []
        )
        lines = []
        for i, r in enumerate(results):
            entity_info = " | ".join(
                f"{n}: {r.get(n, 'N/A')}" for n in entity_names
            )
            if not entity_info:
                entity_info = f"Type: {r.get('document_type', 'N/A')}"
            lines.append(
                f"[{i}] {entity_info} | "
                f"Tipo: {r.get('document_type', 'N/A')} | "
                f"Sección: {r.get('section_title', 'N/A')}\n"
                f"{r.get('text', '')[:500]}"
            )
        return "\n\n".join(lines)

    def _parse_rerank_scores(self, raw: str) -> dict[int, float]:
        """Parse LLM rerank response into index->score mapping."""
        import json

        m = _re.search(r"\[.*\]", raw, _re.DOTALL)
        if not m:
            # Try to parse truncated JSON by closing the array
            m2 = _re.search(r"\[.*", raw, _re.DOTALL)
            if m2:
                truncated = m2.group().rstrip().rstrip(",")
                # Close any open object, then close the array
                if truncated.rstrip().endswith("{"):
                    truncated = truncated.rstrip()[:-1].rstrip(",") + "]"
                elif not truncated.endswith("}"):
                    truncated = truncated + "}" + "]"
                else:
                    truncated = truncated + "]"
                try:
                    scores = json.loads(truncated)
                    return {s["index"]: s["score"] for s in scores}
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"No JSON array in response: {raw[:200]}")
        scores = json.loads(m.group())
        return {s["index"]: s["score"] for s in scores}

    def _apply_scores(
        self, results: list[dict], score_map: dict[int, float], top_k: int
    ) -> list[dict]:
        """Apply rerank scores and return top-K results."""
        for i, r in enumerate(results):
            r["rerank_score"] = score_map.get(i, 0)
        results.sort(key=lambda r: r.get("rerank_score", 0), reverse=True)
        return results[:top_k]

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
        model: str = "claude-haiku-4-5-20251001",
        provider: str = "gemini",
    ) -> list[dict]:
        """Re-rank results using an LLM. Default: Gemini 2.5 Flash, fallback: Haiku.

        Args:
            provider: "gemini" (Gemini 2.5 Flash, default) or "haiku" (Claude Haiku).
        """
        if not results:
            return []
        if provider == "gemini":
            reranked = self._rerank_gemini(query, results, top_k)
            # If Gemini failed (no rerank_score), fall back to Haiku
            if reranked and "rerank_score" not in reranked[0]:
                log.info("Gemini rerank failed, falling back to Haiku")
                return self._rerank_haiku(query, results, top_k, model)
            return reranked
        return self._rerank_haiku(query, results, top_k, model)

    def _rerank_haiku(
        self,
        query: str,
        results: list[dict],
        top_k: int,
        model: str,
    ) -> list[dict]:
        """Re-rank using Claude Haiku."""
        try:
            from anthropic import Anthropic

            client = Anthropic()
            prompt = self._RERANK_PROMPT.format(
                query=query,
                results_block=self._build_rerank_block(results),
            )
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            score_map = self._parse_rerank_scores(resp.content[0].text.strip())
            return self._apply_scores(results, score_map, top_k)
        except Exception as e:
            log.warning("Rerank (haiku) failed, returning original order: %s", e)
            return results[:top_k]

    def _rerank_gemini(
        self,
        query: str,
        results: list[dict],
        top_k: int,
        max_retries: int = 3,
    ) -> list[dict]:
        """Re-rank using Gemini 2.5 Flash with retry on rate limits."""
        try:
            import time

            from google import genai
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            client = genai.Client(api_key=api_key)
            prompt = self._RERANK_PROMPT.format(
                query=query,
                results_block=self._build_rerank_block(results),
            )
            for attempt in range(max_retries):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=2048,
                            temperature=0.0,
                            thinking_config=types.ThinkingConfig(thinking_budget=0),
                        ),
                    )
                    score_map = self._parse_rerank_scores(resp.text.strip())
                    return self._apply_scores(results, score_map, top_k)
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait = 2**attempt * 5  # 5s, 10s, 20s
                        log.info("Gemini rate limited, retrying in %ds...", wait)
                        time.sleep(wait)
                        continue
                    raise
        except Exception as e:
            log.warning("Rerank (gemini) failed, returning original order: %s", e)
            return results[:top_k]

    # -- Absence verification ----------------------------------------------

    def verify_absence(
        self,
        keywords: list[str],
        limit: int = 3,
        **filters,
    ) -> dict:
        """Use FTS to verify whether keywords appear in filtered documents.

        This is a complement to semantic search for negation queries.
        FTS is more reliable for confirming absence because it matches
        exact tokens rather than semantic similarity.

        Returns dict with:
            found (bool): True if any keyword was found
            evidence (list[dict]): Matching chunks (empty if not found)
            keywords_matched (list[str]): Which keywords had hits
        """
        evidence = []
        keywords_matched = []
        for kw in keywords:
            try:
                hits = self.fts_search(
                    kw,
                    limit=limit,
                    apply_precedence=False,
                    **filters,
                )
                if hits:
                    evidence.extend(hits)
                    keywords_matched.append(kw)
            except Exception:
                continue

        # Deduplicate by text hash
        seen = set()
        unique = []
        for h in evidence:
            key = hash(h.get("text", "")[:200])
            if key not in seen:
                seen.add(key)
                unique.append(h)

        return {
            "found": len(unique) > 0,
            "evidence": unique[:limit],
            "keywords_matched": keywords_matched,
        }
