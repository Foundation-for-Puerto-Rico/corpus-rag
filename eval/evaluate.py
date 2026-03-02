"""Evaluation framework for corpus-rag retrieval quality.

Metrics: Precision@K, Recall@K, MRR, NDCG@K.
Hit definition: result matches expected on entity values AND/OR document_type.

Usage:
    python eval/evaluate.py --label baseline --mode hybrid
    python eval/evaluate.py --label rerank_test --mode hybrid --rerank
    python eval/evaluate.py --label rerank_gemini --mode hybrid --rerank --rerank-provider gemini

Dataset format (eval/eval_dataset.json):
    {
      "queries": [
        {
          "id": "q01",
          "query": "your search query here",
          "category": "category_name",
          "difficulty": "easy|medium|hard",
          "expected_document_types": ["document_type_1"],
          "expected_entities": ["EntityValue1", "EntityValue2"]
        }
      ]
    }

Results are saved to eval/results/{label}_{timestamp}.json.
"""
import json
import math
import logging
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# Default DB path — override with CORPUS_DB_PATH env var
DEFAULT_DB_PATH = Path.home() / "corpus-rag" / "data"


def load_dataset(path: str | Path = DATASET_PATH) -> dict:
    """Load evaluation dataset from JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config() -> dict | None:
    """Load corpus_config.yaml if present, to get table name and other settings."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "corpus_config.yaml"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
    except ImportError:
        pass
    return None


def is_hit(result: dict, expected: dict) -> bool:
    """Check if a result is a hit against expected criteria.

    A hit requires:
    - If expected_entities is non-empty: result primary entity must be in expected
      list AND result document_type must be in expected list.
    - If expected_entities is empty: only document_type match is required.

    The field name for entity matching is "entity" by default, but also checks
    "comunidad" for backward compatibility with WCRP-style datasets.
    """
    expected_entities = expected.get("expected_entities", [])
    expected_doc_types = expected.get("expected_document_types", [])

    doc_type_match = result.get("document_type") in expected_doc_types

    if not expected_entities:
        return doc_type_match

    # Check common entity field names
    entity_value = (
        result.get("entity")
        or result.get("comunidad")
        or result.get("community")
        or result.get("city")
        or result.get("department")
    )
    entity_match = entity_value in expected_entities
    return entity_match and doc_type_match


def precision_at_k(results: list[dict], expected: dict, k: int) -> float:
    """Fraction of top-K results that are hits."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if is_hit(r, expected))
    return hits / len(top_k)


def recall_at_k(results: list[dict], expected: dict, k: int) -> float:
    """Fraction of expected items found in top-K results.

    When expected_entities is non-empty, recall = fraction of expected entities
    found in results. When empty, recall = fraction of expected document_types found.
    """
    top_k = results[:k]
    expected_entities = expected.get("expected_entities", [])
    expected_doc_types = expected.get("expected_document_types", [])

    if expected_entities:
        found = set()
        for r in top_k:
            if is_hit(r, expected):
                entity_value = (
                    r.get("entity")
                    or r.get("comunidad")
                    or r.get("community")
                    or r.get("city")
                    or r.get("department")
                )
                if entity_value:
                    found.add(entity_value)
        return len(found) / len(expected_entities) if expected_entities else 0.0
    else:
        found_types = set()
        for r in top_k:
            if r.get("document_type") in expected_doc_types:
                found_types.add(r.get("document_type"))
        return len(found_types) / len(expected_doc_types) if expected_doc_types else 0.0


def mrr(results: list[dict], expected: dict) -> float:
    """Mean Reciprocal Rank: 1/rank of first hit."""
    for i, r in enumerate(results, 1):
        if is_hit(r, expected):
            return 1.0 / i
    return 0.0


def ndcg_at_k(results: list[dict], expected: dict, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Binary relevance: 1 if hit, 0 otherwise.
    """
    top_k = results[:k]
    if not top_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, r in enumerate(top_k):
        rel = 1.0 if is_hit(r, expected) else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all hits come first
    n_hits = sum(1 for r in top_k if is_hit(r, expected))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_hits))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(retriever, dataset_path=DATASET_PATH, mode="hybrid", limit=10,
             rerank=False, rerank_provider="gemini") -> dict:
    """Run full evaluation against the dataset.

    Args:
        retriever: Retriever instance with search methods.
        dataset_path: Path to evaluation dataset JSON.
        mode: Search mode - "hybrid", "semantic", or "fts".
        limit: Number of results per query.
        rerank: Whether to apply re-ranking.
        rerank_provider: "haiku" or "gemini".

    Returns:
        Dict with avg metrics and per-query details.
    """
    dataset = load_dataset(dataset_path)
    per_query = []

    search_fn = {
        "hybrid": retriever.hybrid_search,
        "semantic": retriever.semantic_search,
        "fts": retriever.fts_search,
    }.get(mode, retriever.hybrid_search)

    fetch_limit = limit * 4 if rerank else limit

    for q in dataset["queries"]:
        try:
            results = search_fn(q["query"], limit=fetch_limit)
            if rerank and results:
                results = retriever.rerank(q["query"], results, top_k=limit,
                                           provider=rerank_provider)
        except Exception as e:
            log.warning("Query %s failed: %s", q["id"], e)
            results = []

        metrics = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "p_at_5": precision_at_k(results, q, 5),
            "p_at_10": precision_at_k(results, q, 10),
            "recall_at_10": recall_at_k(results, q, 10),
            "mrr": mrr(results, q),
            "ndcg_at_5": ndcg_at_k(results, q, 5),
            "n_results": len(results),
        }
        per_query.append(metrics)

    n = len(per_query) or 1
    summary = {
        "mode": mode,
        "limit": limit,
        "rerank": rerank,
        "rerank_provider": rerank_provider if rerank else None,
        "n_queries": len(per_query),
        "avg_p_at_5": sum(q["p_at_5"] for q in per_query) / n,
        "avg_p_at_10": sum(q["p_at_10"] for q in per_query) / n,
        "avg_recall_at_10": sum(q["recall_at_10"] for q in per_query) / n,
        "avg_mrr": sum(q["mrr"] for q in per_query) / n,
        "avg_ndcg_at_5": sum(q["ndcg_at_5"] for q in per_query) / n,
        "per_query": per_query,
    }
    return summary


def save_results(metrics: dict, label: str, output_dir: str = "eval/results/") -> Path:
    """Save evaluation results to JSON file."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = out_path / f"{label}_{timestamp}.json"
    filepath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Results saved to %s", filepath)
    return filepath


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate corpus-rag retrieval quality")
    parser.add_argument("--label", required=True, help="Label for this eval run")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "semantic", "fts"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-provider", default="gemini",
                        choices=["haiku", "gemini"],
                        help="LLM provider for re-ranking")
    parser.add_argument("--dataset", default=str(DATASET_PATH),
                        help="Path to eval dataset JSON (default: eval/eval_dataset.json)")
    parser.add_argument("--db-path",
                        default=os.environ.get("CORPUS_DB_PATH", str(DEFAULT_DB_PATH)),
                        help="Path to LanceDB database directory")
    args = parser.parse_args()

    # Validate dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Create eval/eval_dataset.json with your queries. Format:")
        print('  {"queries": [{"id": "q01", "query": "...", "category": "...",')
        print('    "difficulty": "easy|medium|hard",')
        print('    "expected_document_types": ["type1"],')
        print('    "expected_entities": ["Entity1"]}]}')
        sys.exit(1)

    # Load config for table name if available
    config = load_config()
    table_name = None
    if config:
        table_name = config.get("project", {}).get("table_name")

    sys.path.insert(0, ".")
    from retriever import Retriever

    retriever_kwargs = {"db_path": args.db_path}
    if table_name:
        retriever_kwargs["table_name"] = table_name

    ret = Retriever(**retriever_kwargs)
    results = evaluate(ret, dataset_path=dataset_path, mode=args.mode,
                       limit=args.limit, rerank=args.rerank,
                       rerank_provider=args.rerank_provider)
    path = save_results(results, args.label)

    print(f"\n{'='*60}")
    print(f"Evaluation: {args.label} (mode={args.mode}, limit={args.limit})")
    if args.rerank:
        print(f"Re-ranking: {args.rerank_provider}")
    print(f"{'='*60}")
    print(f"  Avg P@5:       {results['avg_p_at_5']:.3f}")
    print(f"  Avg P@10:      {results['avg_p_at_10']:.3f}")
    print(f"  Avg Recall@10: {results['avg_recall_at_10']:.3f}")
    print(f"  Avg MRR:       {results['avg_mrr']:.3f}")
    print(f"  Avg NDCG@5:    {results['avg_ndcg_at_5']:.3f}")
    print(f"\nResults saved to {path}")
