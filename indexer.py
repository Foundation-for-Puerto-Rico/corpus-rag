#!/usr/bin/env python3
"""Config-driven RAG Indexer — Processes markdown files into a searchable
LanceDB vector database with Nomic Embed v1.5 embeddings.

All domain-specific logic (entity maps, document types, skip sections) comes
from a corpus_config.yaml file loaded via config_loader.

Usage:
    python indexer.py                        # Use default corpus_config.yaml
    python indexer.py --config my_config.yaml  # Use custom config
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

from config_loader import CorpusConfig, load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("CORPUS_DB_PATH", str(Path.home() / "corpus-rag" / "data"))
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Chunking thresholds (in estimated tokens)
MAX_SECTION_TOKENS = 1500
MIN_CHUNK_TOKENS = 100
OVERLAP_TOKENS = 125
# Hard cap: Nomic v1.5 context is 8192 tokens, but nomic-bert-2048 uses 2048.
# Truncate to stay well within limits.
MAX_CHUNK_TOKENS = 1800

# Spanish text: ~1.3 tokens per whitespace-delimited word
TOKENS_PER_WORD = 1.3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Section:
    title: str
    level: int           # 1 = #, 2 = ##, 3 = ###, etc.
    content: str         # text content (no header line)
    subsections: list["Section"] = field(default_factory=list)


@dataclass
class Chunk:
    text: str
    entity_values: dict[str, str]     # e.g. {"city": "Boston", "region": "northeast"}
    custom_values: dict[str, str]     # e.g. {"year": "2024"}
    section_title: str
    section_level: int
    content_type: str                 # prose | table | list | mixed
    has_quantitative_data: bool
    chunk_index: int
    source_file: str
    document_type: str = ""
    page_start: int = -1
    page_end: int = -1
    context: str = ""
    is_summary: bool = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    words = text.split()
    return int(len(words) * TOKENS_PER_WORD)


def detect_content_type(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "prose"
    table_lines = sum(1 for l in lines if "|" in l)
    list_lines = sum(1 for l in lines if re.match(r"^[-*\u2022]\s|^\d+[.)]\s", l))
    total = len(lines)
    if table_lines > total * 0.4:
        return "table"
    if list_lines > total * 0.4:
        return "list"
    if table_lines > 0 or list_lines > 0:
        return "mixed"
    return "prose"


def has_quantitative(text: str) -> bool:
    patterns = [
        r"\d+[.,]\d+\s*%",
        r"\d+\s*%",
        r"\$[\d,]+",
        r"\d{1,3}(?:,\d{3})+",
        r"\d+\s*(?:personas|residentes|habitantes|viviendas|hogares|familias)",
    ]
    return any(re.search(p, text) for p in patterns)


def should_skip(title: str, config: CorpusConfig) -> bool:
    """Check if a section should be skipped, delegating to config."""
    return config.should_skip_section(title)


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

_IMG_RE = re.compile(r"^!\[([^\]]*)\]\([^)]+\)\s*$")
_CAPTION_RE = re.compile(r"^Imagen\s+\d+\.")
_PAGENUM_RE = re.compile(r"^(?:[ivxlc]+|\d{1,3})$")


def clean_markdown(text: str) -> str:
    """Strip image tags, duplicate alt text, photo captions, page numbers."""
    lines = text.split("\n")
    out: list[str] = []
    prev_alt: str | None = None

    for line in lines:
        stripped = line.strip()

        # Image tag  ![alt](file)
        m = _IMG_RE.match(stripped)
        if m:
            prev_alt = m.group(1).strip() or None
            continue

        # Duplicate alt-text line immediately after an image
        if prev_alt and stripped:
            if stripped == prev_alt or (
                len(prev_alt) > 20 and prev_alt[:30] in stripped
            ):
                prev_alt = None
                continue
            # Second descriptive line (longer re-description)
            if len(stripped) > 40 and prev_alt[:20] in stripped:
                prev_alt = None
                continue
            prev_alt = None

        # Photo caption lines  "Imagen N. …"
        if _CAPTION_RE.match(stripped):
            continue

        # Bare page numbers / roman numerals
        if _PAGENUM_RE.match(stripped):
            continue

        out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Markdown → Section parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def _parse_flat(lines: list[str]) -> list[Section]:
    """Parse lines where ALL headers are # (flat PDF conversion).

    Strategy: skip leading TOC-like headers (very short / no content),
    then treat every # as a level-2 section.
    """
    sections: list[Section] = []
    cur_title = ""
    cur_lines: list[str] = []
    toc_ended = False

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            # flush previous
            content = "\n".join(cur_lines).strip()
            if cur_title or content:
                if not toc_ended:
                    # TOC entries are headers with little/no following content
                    if estimate_tokens(content) < 30:
                        cur_title = m.group(2).strip()
                        cur_lines = []
                        continue
                    toc_ended = True
                if content:
                    sections.append(Section(title=cur_title, level=2, content=content))
            cur_title = m.group(2).strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    # last section — also capture content in single-heading or headingless docs
    # where toc_ended never triggered (no second heading with enough content)
    content = "\n".join(cur_lines).strip()
    if content and (toc_ended or estimate_tokens(content) >= MIN_CHUNK_TOKENS):
        sections.append(Section(title=cur_title, level=2, content=content))

    return sections


def _parse_hierarchical(lines: list[str]) -> list[Section]:
    """Parse lines that use ## / ### hierarchy.

    Returns a list of level-2 sections, each potentially having level-3+
    subsections stored inside ``subsections``.
    """
    top_sections: list[Section] = []
    cur_main: Section | None = None
    cur_sub: Section | None = None
    cur_lines: list[str] = []

    def _flush_sub():
        nonlocal cur_sub, cur_lines
        content = "\n".join(cur_lines).strip()
        if cur_sub is not None:
            cur_sub.content = content
            if cur_main is not None:
                cur_main.subsections.append(cur_sub)
            cur_sub = None
        elif cur_main is not None and content:
            cur_main.content = content
        cur_lines = []

    def _flush_main():
        nonlocal cur_main
        _flush_sub()
        if cur_main is not None:
            top_sections.append(cur_main)
        cur_main = None

    for line in lines:
        m = _HEADER_RE.match(line)
        if m:
            lvl = len(m.group(1))
            title = m.group(2).strip()
            if lvl <= 2:
                _flush_main()
                cur_main = Section(title=title, level=lvl, content="")
                cur_lines = []
            else:  # ### or ####
                _flush_sub()
                cur_sub = Section(title=title, level=lvl, content="")
                cur_lines = []
        else:
            cur_lines.append(line)

    _flush_main()
    return top_sections


def is_flat_doc(text: str) -> bool:
    h2 = len(re.findall(r"^## ", text, re.MULTILINE))
    return h2 == 0


def parse_document(text: str) -> list[Section]:
    lines = text.split("\n")
    if is_flat_doc(text):
        return _parse_flat(lines)
    return _parse_hierarchical(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into paragraph-based chunks with overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text]

    chunks: list[str] = []
    buf: list[str] = []
    buf_tok = 0

    for para in paras:
        pt = estimate_tokens(para)
        if buf_tok + pt > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            # overlap: keep trailing paragraphs
            overlap_buf: list[str] = []
            ot = 0
            for p in reversed(buf):
                ppt = estimate_tokens(p)
                if ot + ppt > overlap_tokens:
                    break
                overlap_buf.insert(0, p)
                ot += ppt
            buf = overlap_buf
            buf_tok = ot
        buf.append(para)
        buf_tok += pt

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _section_full_text(section: Section) -> str:
    """Reconstruct the full text of a section including its sub-sections."""
    parts = []
    if section.content:
        parts.append(section.content)
    for sub in section.subsections:
        parts.append(f"### {sub.title}\n\n{sub.content}")
    return "\n\n".join(parts)


_PAGE_MARKER_RE = re.compile(r'<!--\s*Page\s+(\d+)\s*-->')


def _extract_page_range(text: str) -> tuple[int, int]:
    """Extract page range from <!-- Page N --> markers in chunk text.

    Returns (-1, -1) if no page markers are found.
    """
    pages = _PAGE_MARKER_RE.findall(text)
    if pages:
        nums = [int(p) for p in pages]
        return (min(nums), max(nums))
    return (-1, -1)


def _build_context_line(entity_values: dict[str, str]) -> str:
    """Build context line from entity values for chunk prefix."""
    context_parts = [f"{k}: {v}" for k, v in entity_values.items() if v]
    return ", ".join(context_parts) + "\n\n" if context_parts else ""


def chunk_sections(
    sections: list[Section],
    entity_values: dict[str, str],
    source_file: str,
    config: CorpusConfig,
) -> list[Chunk]:
    """Hierarchical chunking: ## -> ### -> paragraph splits.

    Uses entity_values dict instead of hardcoded comunidad/municipio/region.
    Delegates skip logic to config.should_skip_section().
    """
    chunks: list[Chunk] = []
    idx = 0
    context_line = _build_context_line(entity_values)

    for sec in sections:
        if should_skip(sec.title, config):
            continue

        full = _section_full_text(sec)
        ft = estimate_tokens(full)

        if ft < MIN_CHUNK_TOKENS:
            continue

        # -- Case 1: section fits in one chunk --
        if ft <= MAX_SECTION_TOKENS:
            header = f"## {sec.title}\n\n" if sec.title else ""
            text = context_line + header + full
            chunks.append(Chunk(
                text=text, entity_values=dict(entity_values),
                custom_values={},
                section_title=sec.title, section_level=sec.level,
                content_type=detect_content_type(full),
                has_quantitative_data=has_quantitative(full),
                chunk_index=idx, source_file=source_file,
            ))
            idx += 1
            continue

        # -- Case 2: split by subsections --
        if sec.subsections:
            # Main section's own content (before subsections)
            if sec.content.strip():
                mc = sec.content.strip()
                mt = estimate_tokens(mc)
                header = f"## {sec.title}\n\n" if sec.title else ""
                if mt <= MAX_SECTION_TOKENS:
                    chunks.append(Chunk(
                        text=context_line + header + mc,
                        entity_values=dict(entity_values),
                        custom_values={},
                        section_title=sec.title,
                        section_level=sec.level,
                        content_type=detect_content_type(mc),
                        has_quantitative_data=has_quantitative(mc),
                        chunk_index=idx, source_file=source_file,
                    ))
                    idx += 1
                else:
                    for pc in _split_paragraphs(mc, MAX_SECTION_TOKENS, OVERLAP_TOKENS):
                        chunks.append(Chunk(
                            text=context_line + header + pc,
                            entity_values=dict(entity_values),
                            custom_values={},
                            section_title=sec.title,
                            section_level=sec.level,
                            content_type=detect_content_type(pc),
                            has_quantitative_data=has_quantitative(pc),
                            chunk_index=idx, source_file=source_file,
                        ))
                        idx += 1

            # Each subsection
            for sub in sec.subsections:
                st = estimate_tokens(sub.content)
                sub_title = f"{sec.title} > {sub.title}"
                header = f"## {sec.title}\n### {sub.title}\n\n"
                if st <= MAX_SECTION_TOKENS:
                    chunks.append(Chunk(
                        text=context_line + header + sub.content,
                        entity_values=dict(entity_values),
                        custom_values={},
                        section_title=sub_title,
                        section_level=sub.level,
                        content_type=detect_content_type(sub.content),
                        has_quantitative_data=has_quantitative(sub.content),
                        chunk_index=idx, source_file=source_file,
                    ))
                    idx += 1
                else:
                    for pc in _split_paragraphs(sub.content, MAX_SECTION_TOKENS, OVERLAP_TOKENS):
                        chunks.append(Chunk(
                            text=context_line + header + pc,
                            entity_values=dict(entity_values),
                            custom_values={},
                            section_title=sub_title,
                            section_level=sub.level,
                            content_type=detect_content_type(pc),
                            has_quantitative_data=has_quantitative(pc),
                            chunk_index=idx, source_file=source_file,
                        ))
                        idx += 1
            continue

        # -- Case 3: no subsections, split by paragraphs --
        header = f"## {sec.title}\n\n" if sec.title else ""
        for pc in _split_paragraphs(full, MAX_SECTION_TOKENS, OVERLAP_TOKENS):
            chunks.append(Chunk(
                text=context_line + header + pc,
                entity_values=dict(entity_values),
                custom_values={},
                section_title=sec.title,
                section_level=sec.level,
                content_type=detect_content_type(pc),
                has_quantitative_data=has_quantitative(pc),
                chunk_index=idx, source_file=source_file,
            ))
            idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Post-processing: merge small, truncate large
# ---------------------------------------------------------------------------

def _truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    words = text.split()
    max_words = int(max_tokens / TOKENS_PER_WORD)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def postprocess_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge adjacent small chunks; truncate oversized ones."""
    if not chunks:
        return chunks

    # 1. Merge adjacent small chunks from the same source_file
    merged: list[Chunk] = []
    buf: Chunk | None = None

    for c in chunks:
        tok = estimate_tokens(c.text)

        # Truncate oversized chunks
        if tok > MAX_CHUNK_TOKENS:
            if buf:
                merged.append(buf)
                buf = None
            c.text = _truncate_text(c.text, MAX_CHUNK_TOKENS)
            merged.append(c)
            continue

        if tok >= MIN_CHUNK_TOKENS * 2:
            # Chunk is big enough on its own
            if buf:
                merged.append(buf)
                buf = None
            merged.append(c)
            continue

        # Small chunk — try to merge with buffer
        if buf is None:
            buf = c
            continue

        # Same document and combined size is reasonable?
        combined_tok = estimate_tokens(buf.text) + tok
        if buf.source_file == c.source_file and combined_tok <= MAX_SECTION_TOKENS:
            buf.text = buf.text + "\n\n" + c.text
            buf.section_title = buf.section_title or c.section_title
            buf.content_type = detect_content_type(buf.text)
            buf.has_quantitative_data = (
                buf.has_quantitative_data or c.has_quantitative_data
            )
        else:
            merged.append(buf)
            buf = c

    if buf:
        merged.append(buf)

    # 2. Re-index chunk_index and extract page numbers
    for i, c in enumerate(merged):
        c.chunk_index = i
        c.page_start, c.page_end = _extract_page_range(c.text)

    return merged


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------

def process_document(
    filepath: str,
    config: CorpusConfig,
    document_type: str = "",
) -> list[Chunk]:
    """Process a single markdown document into chunks.

    Uses config for entity extraction instead of hardcoded community maps.

    Args:
        filepath: Path to the markdown file.
        config: Corpus configuration with entity definitions.
        document_type: Override document type (auto-detected from directory if empty).

    Returns:
        List of Chunk objects with entity_values and custom_values filled.
    """
    stem = Path(filepath).stem

    # Extract entities from filename and directory
    entity_values = config.extract_entities(
        filename=stem,
        directory=Path(filepath).parent.name,
    )

    # Extract custom fields
    custom_values = config.extract_custom_fields(filename=stem)

    raw = Path(filepath).read_text(encoding="utf-8")
    cleaned = clean_markdown(raw)
    sections = parse_document(cleaned)
    flat = is_flat_doc(raw)

    chunks = chunk_sections(sections, entity_values, Path(filepath).name, config)
    chunks = postprocess_chunks(chunks)

    # Apply document-level metadata to all chunks
    for c in chunks:
        c.document_type = document_type
        c.custom_values = dict(custom_values)

    entity_summary = ", ".join(f"{k}={v}" for k, v in entity_values.items())
    log.info(
        "  %-45s  %3d chunks  (%s) [%s] type=%s",
        Path(filepath).name, len(chunks), entity_summary,
        "flat" if flat else "hier", document_type,
    )
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str = "corpus_config.yaml"):
    import numpy as np
    import pandas as pd
    import lancedb
    from sentence_transformers import SentenceTransformer

    config = load_config(config_path)

    log.info("=" * 60)
    log.info("Corpus RAG Indexer — %s", config.project.name)
    log.info("=" * 60)

    table_name = config.project.table_name

    # -- 1. Discover documents from all configured directories --
    all_docs: list[tuple[str, str]] = []  # (filepath, document_type)
    for doc_type, directory in config.docs_dirs().items():
        d = Path(directory)
        if not d.exists():
            log.warning("Directory does not exist, skipping: %s", directory)
            continue
        docs = sorted(d.glob("*.md"))
        log.info("Found %d markdown files in %s (%s)", len(docs), directory, doc_type)
        for doc in docs:
            all_docs.append((str(doc), doc_type))
    log.info("Total documents to process: %d", len(all_docs))

    # -- 2. Parse & chunk --
    all_chunks: list[Chunk] = []
    for filepath, doc_type in all_docs:
        all_chunks.extend(process_document(filepath, config, document_type=doc_type))
    log.info("Total chunks: %d", len(all_chunks))

    # -- 2b. Enrich with contextual embeddings & summary indexing --
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        log.info("ANTHROPIC_API_KEY found — enriching with contexts and summaries")
        try:
            from context_generator import generate_contexts_batch
            from summary_generator import generate_summaries_for_document
        except ImportError:
            log.warning("context_generator/summary_generator not found, skipping enrichment")
            api_key = None

    if api_key:
        from collections import defaultdict
        chunks_by_file: dict[str, list[Chunk]] = defaultdict(list)
        for c in all_chunks:
            chunks_by_file[c.source_file].append(c)

        # Generate contextual descriptions per document
        for i, (source_file, file_chunks) in enumerate(chunks_by_file.items(), 1):
            sample = file_chunks[0]
            doc_path = None
            for fp, dt in all_docs:
                if Path(fp).name == source_file:
                    doc_path = fp
                    break
            doc_text = Path(doc_path).read_text(encoding="utf-8")[:3000] if doc_path else ""

            log.info("  [%d/%d] Contexts for %s (%d chunks)",
                     i, len(chunks_by_file), source_file, len(file_chunks))
            generate_contexts_batch(
                chunks=file_chunks,
                document_text=doc_text,
                document_title=source_file.replace(".md", "").replace("_", " "),
                entity_values=sample.entity_values,
                document_type=sample.document_type,
            )

        # Generate section summaries per document
        summary_chunks: list[Chunk] = []
        for i, (source_file, file_chunks) in enumerate(chunks_by_file.items(), 1):
            sample = file_chunks[0]
            sections_map: dict[str, str] = {}
            for c in file_chunks:
                title = c.section_title or "General"
                if title not in sections_map:
                    sections_map[title] = c.text
                else:
                    sections_map[title] += "\n\n" + c.text

            sections_list = [{"title": t, "text": txt} for t, txt in sections_map.items()
                             if len(txt) > 200]

            if sections_list:
                log.info("  [%d/%d] Summaries for %s (%d sections)",
                         i, len(chunks_by_file), source_file, len(sections_list))
                new_summaries = generate_summaries_for_document(
                    sections=sections_list,
                    entity_values=sample.entity_values,
                    source_file=source_file,
                    document_type=sample.document_type,
                )
                for sc in new_summaries:
                    sc.custom_values = dict(sample.custom_values)
                summary_chunks.extend(new_summaries)

        all_chunks.extend(summary_chunks)
        log.info("After enrichment: %d chunks (%d summaries added)",
                 len(all_chunks), len(summary_chunks))
    else:
        log.info("No ANTHROPIC_API_KEY — skipping context/summary enrichment")

    # -- 3. Embed with Nomic v1.5 --
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    model.max_seq_length = 2048

    texts = [
        f"search_document: {c.context}\n\n{c.text}" if c.context
        else f"search_document: {c.text}"
        for c in all_chunks
    ]
    log.info("Generating embeddings for %d chunks ...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=8,
        normalize_embeddings=True,
    )
    log.info("Embeddings shape: %s", embeddings.shape)

    # -- 4. Build DataFrame --
    records = []
    for chunk, emb in zip(all_chunks, embeddings):
        record = {
            "text": chunk.text,
            "vector": emb.tolist(),
            "section_title": chunk.section_title,
            "section_level": chunk.section_level,
            "content_type": chunk.content_type,
            "has_quantitative_data": chunk.has_quantitative_data,
            "chunk_index": chunk.chunk_index,
            "source_file": chunk.source_file,
            "document_type": chunk.document_type,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "context": chunk.context,
            "is_summary": chunk.is_summary,
        }
        # Add entity columns dynamically
        for name, val in chunk.entity_values.items():
            record[name] = val
        # Add custom field columns dynamically
        for name, val in chunk.custom_values.items():
            record[name] = val
        records.append(record)

    df = pd.DataFrame(records)

    # -- 5. Ingest into LanceDB --
    log.info("Connecting to LanceDB at %s", DB_PATH)
    db = lancedb.connect(DB_PATH)

    existing = db.list_tables()
    table_names = existing.tables if hasattr(existing, "tables") else list(existing)
    if table_name in table_names:
        db.drop_table(table_name)
        log.info("Dropped existing table '%s'", table_name)

    table = db.create_table(table_name, data=df)
    log.info("Created table '%s' with %d rows", table_name, len(df))

    # -- 6. Create full-text search index --
    log.info("Creating FTS index on 'text' column ...")
    table.create_fts_index("text")
    log.info("FTS index created.")

    # -- 7. Summary --
    log.info("")
    log.info("=" * 60)
    log.info("INDEXING COMPLETE")
    log.info("=" * 60)
    log.info("Total chunks        : %d", len(df))
    log.info("Documents processed : %d", len(all_docs))

    # Log entity distributions
    for entity in config.entities:
        if entity.name in df.columns:
            n_unique = df[entity.name].nunique()
            log.info("Unique %-14s: %d", entity.name, n_unique)

    log.info("")
    log.info("Content type distribution:")
    for ct, cnt in df["content_type"].value_counts().items():
        log.info("  %-10s %d", ct, cnt)

    if "document_type" in df.columns:
        log.info("")
        log.info("Document type distribution:")
        for dt, cnt in df["document_type"].value_counts().items():
            log.info("  %-20s %d", dt, cnt)

    log.info("")
    log.info("Done. Database at: %s", DB_PATH)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()
    # Suppress noisy HTTP request logging from httpx/anthropic
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Config-driven RAG Indexer")
    parser.add_argument(
        "--config", default="corpus_config.yaml",
        help="Path to corpus config YAML file (default: corpus_config.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
