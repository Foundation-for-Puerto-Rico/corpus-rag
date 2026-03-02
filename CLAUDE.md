# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this

Config-driven RAG pipeline for any document corpus. All domain structure lives in
`corpus_config.yaml` instead of hardcoded Python constants. An LLM analyzer proposes
the initial config from a corpus sample; the user edits it; the pipeline consumes it.

Forked from WCRP-RAG (Foundation for Puerto Rico's community resilience plans pipeline).

## Commands

```bash
source .venv/bin/activate

# Install
pip install .                              # Core (retriever + server, no PyTorch)
pip install ".[local]"                     # + parser + indexer (PyTorch)
pip install ".[dev]"                       # + pytest

# Tests
python -m pytest tests/ -v                 # All tests (~144)
python -m pytest tests/test_indexer.py -v  # Single file
python -m pytest tests/test_indexer.py::test_chunk_sections_basic -v  # Single test
python -m pytest tests/ -k "flat"          # By keyword

# Index & serve
python indexer.py                          # Full indexation
python reindex.py                          # Incremental (changed files only)
python mcp_server.py                       # MCP server (stdio)
python server.py                           # MCP+REST server (SSE)
```

## Architecture

```
PDFs → parser/ → Markdown → indexer.py (reads config) → LanceDB
                                                           ↑
corpus_config.yaml → config_loader.py → CorpusConfig ──→ retriever.py
                                                       → mcp_server.py → MCP tools
                                                       → server.py → REST + SSE
```

**Core principle:** Nothing domain-specific is in Python code. All entities, document types,
filters, mappings, and MCP instructions come from the YAML config. The `config_loader.py`
is the sole entry point — every module either accepts a `CorpusConfig` argument or calls
`load_config()`.

### Entity extraction order matters

Entities are processed in YAML declaration order, and `mapping` entities can reference
previously resolved entities. Never place a mapping entity above its source:

```yaml
entities:
  - name: "city"            # resolved first
    extract_from: "filename"
  - name: "region"          # resolved second, maps from city
    extract_from: "mapping"
    mapping_source: "city"
```

If a mapping source is unresolved, the derived entity is silently omitted (no error).

### Two embedding paths

- **Indexing** (`indexer.py`): SentenceTransformer (PyTorch), requires `[local]` extra.
  Prefix: `search_document: {context}\n\n{text}`
- **Retrieval** (`retriever.py`): ONNX + tokenizers, no PyTorch. Manual mean-pooling + L2 norm.
  Prefix: `search_query: {query}`

Both use Nomic Embed v1.5. The prefixes must stay consistent — mismatching them breaks
retrieval quality.

### Dynamic LanceDB schema

No fixed schema exists. The DataFrame written at index time has core columns (`text`,
`vector`, `section_title`, etc.) plus **one column per entity and per custom_field** from
config. The Retriever handles missing columns via `build_field_defaults()` (empty strings
for missing string columns, etc.), so older databases survive config additions.

### Chunking cascade

`chunk_sections()` applies in priority order:
1. Section fits whole (≤1500 tokens) → single chunk
2. Has `###` subsections → split by subsection (title becomes `"Parent > Sub"`)
3. Too large, no subsections → paragraph split with 125-token overlap

`postprocess_chunks()` then merges adjacent small chunks (<200 tokens) and truncates
oversized ones (>1800 tokens). Flat documents (no `##` headers) are detected by
`is_flat_doc()` and treat all `#` as level-2.

### Hybrid search deduplication

RRF deduplication key uses the **first configured entity name** as primary component.
If no entities are configured, falls back to `document_type`. Over-fetch factor is `limit * 3`.

### MCP tools are generic, instructions are specific

Five fixed tool signatures regardless of corpus. Domain specificity comes from
`build_instructions()` which interpolates `{project_name}`, `{entity_names}`,
`{filter_list}`, `{document_types}`, `{entity_count}` into `config.mcp.instructions`.

### server.py is a thin adapter

It imports the configured `mcp` FastMCP instance and `Retriever` from `mcp_server.py`,
then adds REST routes via `mcp.custom_route()`. MCP SSE and REST share the same process
and Retriever instance. API key auth applies only to REST endpoints.

### Three optional LLM enrichments (all gated on ANTHROPIC_API_KEY)

1. **Contextual embeddings** (`context_generator.py`): 2-3 sentence chunk context, prepended at embed time
2. **Summary indexing** (`summary_generator.py`): Per-section summaries as separate chunks, demoted 30% in retrieval
3. **Re-ranking** (`retriever.py`): Gemini 2.5 Flash (preferred) or Claude Haiku, scores 0-10

All use JSON file caches keyed by MD5 of content, with atomic writes (`os.replace`).

### Reindexer compound keys

Manifest keys are `"document_type/filename"` to avoid collision when the same filename
exists in multiple document type directories. LanceDB deletion queries use both fields.

## Testing patterns

All tests use `tmp_path` fixtures — **no real DB, no real model, no API calls**.

- **config_loader**: Write YAML to `tmp_path`, call `load_config()`. Pure unit tests.
- **indexer**: Write markdown to `tmp_path`, call `process_document()`. Chunking logic is pure Python.
- **retriever**: Uses `object.__new__(Retriever)` to skip `__init__` (which connects to LanceDB),
  then manually sets `_config`, `_valid_filters`, etc. Tests search builders in isolation.
- **mcp_server**: Uses `monkeypatch.setitem(sys.modules, ...)` to inject `MagicMock` for
  `mcp`, `lancedb`, `dotenv` **before** importing the module. `tool()` decorator is patched
  as passthrough. Module is deleted from `sys.modules` between tests to force reload.
- **integration**: Full `load_config` → `process_document` pipeline on in-memory files.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORPUS_CONFIG_PATH` | `corpus_config.yaml` | Path to config file |
| `CORPUS_DB_PATH` | `~/corpus-rag/data` | LanceDB directory |
| `CORPUS_ONNX_MODEL_DIR` | `~/corpus-rag/models/nomic-embed-text-v1.5/` | ONNX model |
| `CORPUS_MANIFEST_PATH` | `~/corpus-rag/.file_manifest.json` | Reindex manifest |
| `CORPUS_API_KEY` | (none) | REST API auth key |
| `CORPUS_PORT` | `8080` | Server port |
| `ANTHROPIC_API_KEY` | (none) | For re-ranking, context, summaries |
| `GEMINI_API_KEY` | (none) | For Gemini re-ranking (not in pyproject.toml — install `google-genai` manually) |

## Conventions

- Python 3.10+ (type hints with `str | None`)
- Embedding prefixes: `search_document:` for indexing, `search_query:` for searching
- Chunking: hierarchical by markdown sections (## → ### → paragraphs)
- Dependency split by role: core (no PyTorch) vs `[local]` (PyTorch + parser)
- Config sections: project, entities, document_types, custom_fields, filters, skip_sections, mcp
