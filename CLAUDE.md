# CLAUDE.md

## What is this

Config-driven RAG pipeline. All domain structure lives in `corpus_config.yaml`
instead of hardcoded Python constants. An LLM analyzer proposes the initial
config from a corpus sample; the user edits it; the pipeline consumes it.

## Commands

```bash
source .venv/bin/activate

# Analyze corpus and generate config
python analyze_corpus.py docs/type_a/ docs/type_b/

# Index documents
python indexer.py                    # Full indexation
python reindex.py                    # Incremental
python reindex.py --full             # Full re-index
python reindex.py --dry-run          # Preview

# Servers
python mcp_server.py                 # MCP server (stdio)
python server.py                     # MCP+REST server (SSE)

# Tests
python -m pytest tests/ -v

# Utilities
python download_onnx_model.py        # Download ONNX model (once)
```

## Key files

- `corpus_config.yaml` — All domain structure (entities, document types, filters)
- `config_loader.py` — Reads and validates the config
- `analyze_corpus.py` — LLM-powered corpus analyzer that generates initial config
- `indexer.py` — Config-driven chunking + embedding → LanceDB
- `retriever.py` — Semantic/FTS/hybrid search + re-ranking
- `mcp_server.py` — MCP server with tools generated from config
- `server.py` — MCP+REST server (SSE transport for remote access)

## Architecture

```
PDFs → parser/ → Markdown
analyze_corpus.py (LLM) → corpus_config.yaml (user edits)
indexer.py (reads config) → LanceDB
mcp_server.py (reads config) → Dynamic MCP tools
server.py → SSE + REST endpoints
```
