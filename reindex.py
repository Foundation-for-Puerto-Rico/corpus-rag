#!/usr/bin/env python3
"""Corpus RAG Re-indexer — Incremental or full re-indexation into LanceDB.

Config-driven: reads corpus_config.yaml for document directories, table name,
and entity definitions.

Usage:
    python reindex.py            # Incremental (only changed files)
    python reindex.py --full     # Full re-index (drop & recreate)
    python reindex.py --dry-run  # Show what would change, no modifications
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config_loader import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.environ.get("CORPUS_CONFIG_PATH", "corpus_config.yaml")

DB_PATH = os.environ.get("CORPUS_DB_PATH", str(Path.home() / "corpus-rag" / "data"))
MANIFEST_PATH = os.environ.get(
    "CORPUS_MANIFEST_PATH", str(Path.home() / "corpus-rag" / ".file_manifest.json")
)
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Lazy-loaded config — initialized on first use via _get_config()
_config = None
_docs_dirs: dict[str, str] | None = None
_table_name: str | None = None


def _get_config():
    """Load config lazily on first use."""
    global _config, _docs_dirs, _table_name
    if _config is None:
        _config = load_config(CONFIG_PATH)
        _docs_dirs = _config.docs_dirs()
        _table_name = _config.project.table_name
    return _config, _docs_dirs, _table_name


def _reset_config():
    """Force config reload on next _get_config() call."""
    global _config, _docs_dirs, _table_name
    _config = None
    _docs_dirs = None
    _table_name = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: str = MANIFEST_PATH) -> dict:
    """Load manifest from disk or return empty dict."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: str = MANIFEST_PATH) -> None:
    """Save manifest to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# File scanning & diffing
# ---------------------------------------------------------------------------

def compute_md5(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def scan_docs(
    docs_dir: str | None = None,
    docs_dirs: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Scan docs directory(ies) and return {key: {mtime, md5, document_type, dir, filename}}.

    Keys use the format ``document_type/filename`` to avoid collisions when
    the same filename exists in multiple directories.

    Call with a single ``docs_dir`` for backward compatibility, or with
    ``docs_dirs`` (a {document_type: path} dict) to scan multiple directories.
    """
    if docs_dirs is None:
        _, cfg_docs_dirs, _ = _get_config()
        if docs_dir is None:
            docs_dir = list(cfg_docs_dirs.values())[0] if cfg_docs_dirs else "."
        first_type = next(iter(cfg_docs_dirs), "default")
        docs_dirs = {first_type: docs_dir}

    files: dict[str, dict] = {}
    for doc_type, directory in docs_dirs.items():
        d = Path(directory)
        if not d.exists():
            log.warning("Directory does not exist, skipping: %s", directory)
            continue
        for p in sorted(d.glob("*.md")):
            key = f"{doc_type}/{p.name}"
            files[key] = {
                "mtime": p.stat().st_mtime,
                "md5": compute_md5(str(p)),
                "document_type": doc_type,
                "dir": directory,
                "filename": p.name,
            }
    return files


def diff_files(
    current: dict[str, dict],
    manifest: dict,
) -> tuple[set[str], set[str], set[str]]:
    """Compare current files against manifest.

    Returns:
        (new_files, modified_files, deleted_files) — each a set of filenames.
    """
    manifest_files = manifest.get("files", {})
    current_names = set(current.keys())
    manifest_names = set(manifest_files.keys())

    new = current_names - manifest_names
    deleted = manifest_names - current_names

    modified = set()
    for name in current_names & manifest_names:
        cur = current[name]
        prev = manifest_files[name]
        # Quick check: mtime unchanged → skip MD5
        if cur["mtime"] == prev.get("mtime"):
            continue
        # mtime changed — verify with MD5
        if cur["md5"] != prev.get("md5"):
            modified.add(name)

    return new, modified, deleted


# ---------------------------------------------------------------------------
# Incremental re-indexing
# ---------------------------------------------------------------------------

def incremental_reindex(dry_run: bool = False) -> None:
    """Re-index only changed files."""
    config, docs_dirs, table_name = _get_config()

    log.info("=" * 60)
    log.info("Corpus RAG Incremental Re-indexer — %s", config.project.name)
    log.info("=" * 60)

    # 1. Load manifest
    manifest = load_manifest()
    if not manifest:
        log.info("No manifest found — all files will be indexed.")

    # 2. Scan current docs (all configured directories)
    active_dirs = {dt: d for dt, d in docs_dirs.items() if Path(d).exists()}
    log.info("Scanning %d directory(ies) ...", len(active_dirs))
    for dt, d in active_dirs.items():
        log.info("  %s: %s", dt, d)
    current = scan_docs(docs_dirs=active_dirs)
    log.info("Found %d markdown files.", len(current))

    # 3. Diff
    new, modified, deleted = diff_files(current, manifest)

    if not new and not modified and not deleted:
        log.info("No changes detected. Database is up to date.")
        return

    log.info("")
    log.info("Changes detected:")
    if new:
        log.info("  New:      %d files — %s", len(new), ", ".join(sorted(new)))
    if modified:
        log.info("  Modified: %d files — %s", len(modified), ", ".join(sorted(modified)))
    if deleted:
        log.info("  Deleted:  %d files — %s", len(deleted), ", ".join(sorted(deleted)))

    if dry_run:
        log.info("")
        log.info("DRY RUN — no changes made.")
        return

    # 4. Open database
    import lancedb
    db = lancedb.connect(DB_PATH)
    table = db.open_table(table_name)

    # 5. Delete chunks for removed/modified files
    to_remove = deleted | modified
    for key in sorted(to_remove):
        # key format: "document_type/filename"
        parts = key.split("/", 1)
        if len(parts) == 2:
            doc_type, fname = parts
            log.info("  Removing chunks for: %s (%s)", fname, doc_type)
            table.delete(f"source_file = '{fname}' AND document_type = '{doc_type}'")
        else:
            # Legacy key (plain filename) — fallback
            log.info("  Removing chunks for: %s", key)
            table.delete(f"source_file = '{key}'")

    # 6. Index new/modified files
    to_add = new | modified
    if to_add:
        from indexer import process_document
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model ...")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        model.max_seq_length = 2048

        all_chunks = []
        for key in sorted(to_add):
            file_info = current[key]
            doc_type = file_info.get("document_type", "")
            file_dir = file_info.get("dir", ".")
            filename = file_info.get("filename", key.split("/", 1)[-1])
            filepath = str(Path(file_dir) / filename)
            chunks = process_document(filepath, config, document_type=doc_type)
            all_chunks.extend(chunks)

        if all_chunks:
            log.info("Generating embeddings for %d chunks ...", len(all_chunks))
            texts = [
                f"search_document: {c.context}\n\n{c.text}" if c.context
                else f"search_document: {c.text}"
                for c in all_chunks
            ]
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                batch_size=8,
                normalize_embeddings=True,
            )

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

            import pandas as pd
            df = pd.DataFrame(records)
            table.add(df)
            log.info("Added %d chunks to database.", len(df))

    # 7. Rebuild FTS index
    log.info("Rebuilding FTS index ...")
    table.create_fts_index("text", replace=True)
    log.info("FTS index rebuilt.")

    # 8. Update manifest
    manifest_files = {}
    # Get chunk counts per (source_file, document_type) from the updated table
    all_df = table.to_pandas()
    chunk_counts = all_df.groupby(["source_file", "document_type"]).size().to_dict()

    for key, info in current.items():
        doc_type = info.get("document_type", "")
        filename = info.get("filename", key.split("/", 1)[-1])
        manifest_files[key] = {
            "mtime": info["mtime"],
            "md5": info["md5"],
            "chunk_count": chunk_counts.get((filename, doc_type), 0),
            "document_type": doc_type,
        }

    new_manifest = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    }
    save_manifest(new_manifest)
    log.info("Manifest saved to %s", MANIFEST_PATH)

    # 9. Summary
    log.info("")
    log.info("=" * 60)
    log.info("RE-INDEX COMPLETE")
    log.info("=" * 60)
    log.info("Files added:    %d", len(new))
    log.info("Files modified: %d", len(modified))
    log.info("Files deleted:  %d", len(deleted))
    log.info("Total chunks:   %d", len(all_df))
    log.info("Total files:    %d", len(manifest_files))


# ---------------------------------------------------------------------------
# Full re-indexing
# ---------------------------------------------------------------------------

def full_reindex() -> None:
    """Full re-index — delegates to indexer.main() then creates manifest."""
    config, docs_dirs, table_name = _get_config()

    log.info("=" * 60)
    log.info("Corpus RAG Full Re-indexer — %s", config.project.name)
    log.info("=" * 60)

    from indexer import main as indexer_main
    indexer_main(CONFIG_PATH)

    # Create fresh manifest
    log.info("")
    log.info("Creating manifest ...")
    active_dirs = {dt: d for dt, d in docs_dirs.items() if Path(d).exists()}
    current = scan_docs(docs_dirs=active_dirs)

    import lancedb
    db = lancedb.connect(DB_PATH)
    table = db.open_table(table_name)
    all_df = table.to_pandas()
    chunk_counts = all_df.groupby(["source_file", "document_type"]).size().to_dict()

    manifest_files = {}
    for key, info in current.items():
        doc_type = info.get("document_type", "")
        filename = info.get("filename", key.split("/", 1)[-1])
        manifest_files[key] = {
            "mtime": info["mtime"],
            "md5": info["md5"],
            "chunk_count": chunk_counts.get((filename, doc_type), 0),
            "document_type": doc_type,
        }

    manifest = {
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "files": manifest_files,
    }
    save_manifest(manifest)
    log.info("Manifest saved to %s", MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CONFIG_PATH

    parser = argparse.ArgumentParser(description="Corpus RAG Re-indexer")
    parser.add_argument(
        "--full", action="store_true",
        help="Full re-index: drop and recreate the entire database",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without modifying anything",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help="Path to corpus_config.yaml (default: %(default)s)",
    )
    args = parser.parse_args()

    # Reload config if a different path was provided via CLI
    if args.config != CONFIG_PATH:
        CONFIG_PATH = args.config
        _reset_config()

    if args.full:
        if args.dry_run:
            log.warning("--dry-run is ignored with --full (full re-index always modifies data)")
        full_reindex()
    else:
        incremental_reindex(dry_run=args.dry_run)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
