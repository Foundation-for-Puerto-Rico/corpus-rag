# corpus-rag

Turn a collection of PDF documents into an intelligent search system. Ask questions in plain language and get relevant answers from your documents.

## What it does

1. **Reads your PDFs** and converts them to text
2. **Learns the structure** of your documents (types, organization, metadata)
3. **Indexes the content** for fast retrieval
4. **Answers questions** using semantic search (understands meaning, not just exact keywords)

## Requirements

- Python 3.10+
- Anthropic API key (optional — improves search quality with re-ranking and contextual embeddings)

## Getting started

We recommend using [Claude Code](https://claude.ai/code) to set up and work with this project. Clone the repo, open it in Claude Code, and ask it to help you through each step below.

### 1. Install

```bash
git clone https://github.com/Foundation-for-Puerto-Rico/corpus-rag.git
cd corpus-rag

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install ".[local]"       # Includes PDF parser and indexer
python download_onnx_model.py  # Download embedding model (once)
```

### 2. Prepare your documents

Place your PDF files in one or more folders:

```
my-documents/
├── reports/
│   ├── report_2023.pdf
│   └── report_2024.pdf
└── plans/
    └── strategic_plan.pdf
```

### 3. Parse PDFs to text

```bash
python run_parser.py my-documents/reports/ --output markdown/reports/
python run_parser.py my-documents/plans/   --output markdown/plans/
```

### 4. Generate configuration

The system analyzes your documents and creates a configuration file automatically:

```bash
python analyze_corpus.py markdown/reports/ markdown/plans/
```

This generates `corpus_config.yaml` — the file that tells the pipeline how your documents are organized. You can open it and adjust it if needed.

### 5. Index

```bash
python indexer.py
```

### 6. Search

Start the server and connect it to Claude Code or any MCP client:

```bash
python mcp_server.py
```

## Project overview

| File | What it does |
|------|-------------|
| `run_parser.py` | Converts PDFs to Markdown |
| `analyze_corpus.py` | Analyzes your documents and generates `corpus_config.yaml` |
| `corpus_config.yaml` | Defines your document structure (auto-generated or manual) |
| `indexer.py` | Processes documents and builds the search database |
| `reindex.py` | Updates only the documents that changed |
| `mcp_server.py` | Search server for Claude Code (MCP stdio) |
| `server.py` | Web server with REST API (for remote access) |

See `CLAUDE.md` for detailed architecture and development documentation.

## License

MIT
