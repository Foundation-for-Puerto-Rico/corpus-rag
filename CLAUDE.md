# CLAUDE.md

## What is this

Config-driven RAG pipeline for any document corpus. All domain structure lives in
`corpus_config.yaml` instead of hardcoded Python constants. An LLM analyzer proposes
the initial config from a corpus sample; the user edits it; the pipeline consumes it.

Forked from WCRP-RAG (Foundation for Puerto Rico's community resilience plans pipeline).

## Project structure

```
corpus-rag/
├── CLAUDE.md, pyproject.toml, .gitignore
│
├── # --- Config ---
├── corpus_config.yaml          # Domain structure (user-created or LLM-generated)
├── corpus_config.template.yaml # Documented template with examples
├── config_loader.py            # Reads, validates, provides CorpusConfig dataclass
├── analyze_corpus.py           # LLM analyzes sample docs → generates config
│
├── # --- Core RAG pipeline ---
├── indexer.py                  # Config-driven: chunk + embed → LanceDB
├── reindex.py                  # Incremental re-indexation (manifest-based)
├── retriever.py                # Semantic/FTS/hybrid search + re-ranking
├── context_generator.py        # Contextual embeddings (Claude Haiku)
├── summary_generator.py        # Summary indexing (Claude Haiku)
│
├── # --- Servers ---
├── mcp_server.py               # MCP server (stdio) with dynamic tools from config
├── server.py                   # MCP+REST server (SSE, deploy)
│
├── # --- Utilities ---
├── download_onnx_model.py      # Download ONNX model from HuggingFace
├── rebuild_fts.py              # Rebuild FTS index (Tantivy, Linux)
├── run_parser.py               # PDF parser entry point
├── parser/                     # PDF extraction modules (domain-agnostic)
│
├── # --- Deploy & eval ---
├── deploy/deploy.sh            # Deploy to GCP VM
├── eval/evaluate.py            # P@K, Recall, MRR, NDCG metrics
│
├── tests/                      # Unit + integration tests
├── data/                       # LanceDB (gitignored)
└── models/                     # ONNX model (gitignored)
```

## Architecture

```
1. PDFs → parser/ → Markdown files
2. analyze_corpus.py (LLM) → corpus_config.yaml (user edits)
3. indexer.py (reads config) → LanceDB
4. mcp_server.py (reads config) → Dynamic MCP tools
5. server.py → SSE + REST endpoints for remote access
```

The key difference from traditional RAG pipelines: **nothing domain-specific is in the code**.
All entities, document types, filters, mappings, and MCP instructions come from the YAML config.

## Commands

```bash
source .venv/bin/activate

# --- Setup ---
pip install .                              # Core deps (retriever, server)
pip install ".[local]"                     # + parser + indexer (PyTorch)
pip install ".[dev]"                       # + pytest
python download_onnx_model.py             # Download ONNX model (once)

# --- Corpus analysis ---
python analyze_corpus.py docs/reports/ docs/memos/   # LLM generates config
# Or: cp corpus_config.template.yaml corpus_config.yaml  # Manual config

# --- Indexing ---
python indexer.py                          # Full indexation
python reindex.py                          # Incremental (changed files only)
python reindex.py --full                   # Full re-index
python reindex.py --dry-run                # Preview changes

# --- Servers ---
python mcp_server.py                       # MCP server (stdio, for Claude Code)
python server.py                           # MCP+REST server (SSE, for remote)

# --- Tests ---
python -m pytest tests/ -v                 # All tests (144 tests)

# --- Deploy ---
CORPUS_API_KEY=your-key ./deploy/deploy.sh # Deploy to GCP VM
```

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
| `GEMINI_API_KEY` | (none) | For Gemini re-ranking (preferred) |

## The corpus_config.yaml

All domain structure lives here. See `corpus_config.template.yaml` for a documented template.

Key sections:
- **project** — name, description, language, table_name
- **entities** — taxonomy dimensions (extracted from filename/directory/content/mapping)
- **document_types** — categories with directories and precedence boosts
- **custom_fields** — additional per-document metadata
- **filters** — fields exposed in search
- **skip_sections** — boilerplate to exclude from chunking
- **mcp** — server name and instructions for LLM consumers

## Stack

- **Embedding:** Nomic Embed v1.5 (ONNX for retrieval, SentenceTransformer for indexing)
- **Vector DB:** LanceDB with FTS (Tantivy)
- **Search:** Semantic, FTS, hybrid (RRF fusion k=60)
- **Re-ranking:** Gemini 2.5 Flash (default) with Claude Haiku fallback
- **Transport:** MCP stdio + SSE via FastMCP
- **Parser:** PyMuPDF, Tesseract OCR, pdf2image

## Testing

144 tests across 6 test files. All use `tmp_path` fixtures — no real DB or model needed.

```bash
python -m pytest tests/ -v
```

## Conventions

- Python 3.10+ (type hints with `str | None`)
- Embedding prefixes: `search_document:` for indexing, `search_query:` for searching
- Spanish text tokenization: ~1.3 tokens/word
- Chunking: hierarchical by markdown sections (## → ### → paragraphs)
