"""Generate contextual descriptions for chunks using Claude Haiku.

Produces short (2-3 sentence, max 75 word) descriptions that situate each
chunk within its source document.  Results are cached by MD5 of the first
500 characters of chunk text.
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

from anthropic import Anthropic

# Shared client instance (thread-safe — httpx uses connection pooling)
_shared_client: Anthropic | None = None
_client_lock = threading.Lock()

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 150

CONTEXT_CACHE_PATH = os.environ.get(
    "CORPUS_CONTEXT_CACHE",
    os.path.join(os.path.expanduser("~"), ".corpus-rag", ".context_cache.json"),
)

CONTEXT_PROMPT = """\
You are an expert assistant analyzing documents for a RAG system.

Given the following fragment from a document, generate a contextual description \
of 2-3 sentences (maximum 75 words) that situates this fragment within the \
complete document. The description should help a search system understand what \
the fragment is about without repeating its textual content.

Document: {document_title}
Metadata: {metadata}
Document type: {document_type}

Document excerpt (for general context):
{document_excerpt}

Fragment to contextualize:
{chunk_text}

Respond ONLY with the contextual description, no prefixes or explanations."""


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(chunk_text: str) -> str:
    """Return MD5 hex digest of the first 500 chars of *chunk_text*."""
    return hashlib.md5(chunk_text[:500].encode("utf-8")).hexdigest()


def load_context_cache(path: str = CONTEXT_CACHE_PATH) -> dict:
    """Load the context cache from *path*. Return empty dict on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_context_cache(cache: dict, path: str = CONTEXT_CACHE_PATH) -> None:
    """Persist *cache* dict as JSON to *path*, creating parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(cache)  # shallow copy — safe to serialize while writers mutate original
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Single-chunk context generation
# ---------------------------------------------------------------------------

def _build_metadata_string(entity_values: dict[str, str]) -> str:
    """Build a human-readable metadata string from entity values."""
    if not entity_values:
        return "N/A"
    return ", ".join(f"{k}: {v}" for k, v in entity_values.items())


def generate_context(
    chunk_text: str,
    document_title: str,
    document_excerpt: str,
    entity_values: dict[str, str],
    document_type: str,
    cache: dict | None = None,
) -> str:
    """Call Claude Haiku to produce a contextual description for *chunk_text*.

    If *cache* is provided and already contains a result for this chunk, the
    cached value is returned without an API call.  On any error (missing API
    key, network failure, etc.) returns an empty string and logs a warning.
    """
    key = _cache_key(chunk_text)

    if cache is not None and key in cache:
        return cache[key]

    try:
        global _shared_client
        if _shared_client is None:
            with _client_lock:
                if _shared_client is None:
                    _shared_client = Anthropic()
        prompt = CONTEXT_PROMPT.format(
            document_title=document_title,
            metadata=_build_metadata_string(entity_values),
            document_type=document_type,
            document_excerpt=document_excerpt[:2000],
            chunk_text=chunk_text[:2000],
        )
        response = _shared_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        context = response.content[0].text.strip()

        if cache is not None:
            cache[key] = context

        return context

    except Exception as exc:
        logger.warning("Context generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Batch context generation
# ---------------------------------------------------------------------------

def generate_contexts_batch(
    chunks: list,
    document_text: str,
    document_title: str,
    entity_values: dict[str, str],
    document_type: str,
    cache_path: str = CONTEXT_CACHE_PATH,
) -> list:
    """Generate contextual descriptions for a list of Chunk objects.

    Sets the ``.context`` attribute on each chunk in-place and returns the
    list.  Uses a persistent JSON cache at *cache_path*.
    The Anthropic SDK handles 429 rate limiting with automatic retry/backoff.
    """
    cache = load_context_cache(cache_path)
    # Use the first ~3000 chars of the full document as excerpt
    document_excerpt = document_text[:3000]

    for chunk in chunks:
        context = generate_context(
            chunk_text=chunk.text,
            document_title=document_title,
            document_excerpt=document_excerpt,
            entity_values=entity_values,
            document_type=document_type,
            cache=cache,
        )
        chunk.context = context

    save_context_cache(cache, cache_path)
    return chunks
