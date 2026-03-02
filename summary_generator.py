#!/usr/bin/env python3
"""Summary generator for corpus-rag summary indexing.

Uses Claude Haiku to generate section-level summaries of documents.
Summaries are cached via MD5 hashing.
The Anthropic SDK handles 429 rate limiting with automatic retry/backoff.
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from anthropic import Anthropic
from indexer import Chunk

# Shared client instance (thread-safe — httpx uses connection pooling)
_shared_client: Anthropic | None = None
_client_lock = threading.Lock()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CACHE_PATH = os.environ.get(
    "CORPUS_SUMMARY_CACHE",
    str(Path.home() / ".corpus-rag" / ".summary_cache.json"),
)
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 300

SUMMARY_PROMPT = """\
You are a document analyst for a RAG system.

Summarize the following section from a document.

Metadata: {metadata}

Section title: {section_title}

Section text:
{section_text}

Instructions:
- Summarize in a maximum of 200 words.
- Capture key findings, proposed actions, and important quantitative data.
- Write in a technical, third-person tone.
- Do not include headings or bullet points; write in a continuous paragraph.
"""

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(section_text: str) -> str:
    """Generate MD5 cache key from first 500 chars of section text."""
    snippet = section_text[:500]
    return hashlib.md5(snippet.encode("utf-8")).hexdigest()


def load_summary_cache(path: str | None = None) -> dict:
    """Load summary cache from JSON file. Returns empty dict if missing."""
    path = path or DEFAULT_CACHE_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_summary_cache(cache: dict, path: str | None = None) -> None:
    """Save summary cache to JSON file, creating parent dirs if needed."""
    path = path or DEFAULT_CACHE_PATH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(cache)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(p))


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _build_metadata_string(entity_values: dict[str, str]) -> str:
    """Build a human-readable metadata string from entity values."""
    if not entity_values:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in entity_values.items())


def generate_section_summary(
    section_text: str,
    section_title: str,
    entity_values: dict[str, str],
    cache: dict | None = None,
) -> str:
    """Generate a summary for a document section using Claude Haiku.

    If a cache dict is provided and contains a hit, returns the cached value
    without calling the API. On API errors, logs a warning and returns "".
    """
    key = _cache_key(section_text)

    if cache is not None and key in cache:
        return cache[key]

    prompt = SUMMARY_PROMPT.format(
        metadata=_build_metadata_string(entity_values),
        section_title=section_title,
        section_text=section_text,
    )

    try:
        global _shared_client
        if _shared_client is None:
            with _client_lock:
                if _shared_client is None:
                    _shared_client = Anthropic()
        response = _shared_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.content[0].text.strip()
    except Exception as e:
        logger.warning("Error generating summary for '%s': %s", section_title, e)
        return ""

    if cache is not None:
        cache[key] = summary

    return summary


# ---------------------------------------------------------------------------
# Chunk creation
# ---------------------------------------------------------------------------


def create_summary_chunk(
    summary_text: str,
    section_title: str,
    entity_values: dict[str, str],
    source_file: str,
    document_type: str,
) -> Chunk:
    """Create a Chunk marked as a summary for summary indexing."""
    return Chunk(
        text=summary_text,
        entity_values=dict(entity_values),
        custom_values={},
        section_title=section_title,
        section_level=2,
        content_type="summary",
        has_quantitative_data=False,
        chunk_index=-1,
        source_file=source_file,
        document_type=document_type,
        is_summary=True,
    )


# ---------------------------------------------------------------------------
# Document-level orchestration
# ---------------------------------------------------------------------------


def generate_summaries_for_document(
    sections: list[dict],
    entity_values: dict[str, str],
    source_file: str,
    document_type: str,
    cache_path: str | None = None,
) -> list[Chunk]:
    """Generate summary chunks for each section in a document.

    Args:
        sections: list of dicts with keys 'title' and 'text'.
        entity_values: dict of entity name to value (e.g. {"city": "Boston"}).
        source_file: original markdown filename.
        document_type: e.g. 'report', 'plan', etc.
        cache_path: path to the summary cache JSON file.

    Returns:
        List of Chunk objects with is_summary=True.
    """
    cache = load_summary_cache(cache_path)
    chunks: list[Chunk] = []

    for section in sections:
        title = section.get("title", "")
        text = section.get("text", "")
        if not text.strip():
            continue

        summary = generate_section_summary(
            section_text=text,
            section_title=title,
            entity_values=entity_values,
            cache=cache,
        )
        if not summary:
            continue

        chunk = create_summary_chunk(
            summary_text=summary,
            section_title=title,
            entity_values=entity_values,
            source_file=source_file,
            document_type=document_type,
        )
        chunks.append(chunk)

    save_summary_cache(cache, cache_path)
    return chunks
