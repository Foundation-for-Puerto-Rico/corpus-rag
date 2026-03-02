#!/usr/bin/env python3
"""Rebuild the LanceDB FTS (Tantivy) index.

The FTS index is platform-specific and not portable between macOS and Linux.
Run this script after transferring the LanceDB data to a new machine.

Usage:
    python rebuild_fts.py
"""

import os
import sys
from pathlib import Path

import lancedb

DB_PATH = os.environ.get("WCRP_DB_PATH", str(Path.home() / "wcrp-rag" / "data"))
TABLE_NAME = "wcrp_chunks"


def rebuild_fts():
    print(f"Opening LanceDB at {DB_PATH}")
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)

    row_count = table.count_rows()
    print(f"Table '{TABLE_NAME}' has {row_count} rows")

    print("Rebuilding FTS index on 'text' column...")
    table.create_fts_index("text", replace=True)
    print("FTS index rebuilt successfully")

    # Sanity check
    print("Running sanity check query...")
    results = table.search("inundaciones", query_type="fts").limit(3).to_pandas()
    if results.empty:
        print("WARNING: Sanity check returned no results!", file=sys.stderr)
        sys.exit(1)
    print(f"Sanity check passed: {len(results)} results for 'inundaciones'")
    for _, row in results.iterrows():
        print(f"  - {row['comunidad']}: {row['text'][:80]}...")


if __name__ == "__main__":
    rebuild_fts()
