"""JSON output module for ParsedDocument serialization.

Provides JSON serialization with provenance schema for programmatic analysis.
Preserves Spanish characters and includes full metadata and statistics.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from parser.output.document import ParsedDocument


PARSER_VERSION = "1.0.0"


def document_to_dict(doc: ParsedDocument) -> Dict[str, Any]:
    """Convert ParsedDocument to JSON-serializable dict with provenance.

    Args:
        doc: ParsedDocument to serialize.

    Returns:
        Dictionary ready for JSON serialization.
    """
    # Collect all tables with provenance
    tables: List[Dict[str, Any]] = []
    for page in doc.pages:
        for idx, table in enumerate(page.tables):
            tables.append({
                "id": f"p{table.page_number}_t{table.table_index}",
                "page_number": table.page_number,
                "table_index": table.table_index,
                "bbox": list(table.bbox),
                "is_borderless": table.is_borderless,
                "needs_review": table.needs_review,
                "headers": table.header_names,
                "rows": table.rows,
                "row_count": table.row_count,
                "col_count": table.col_count,
            })

    # Collect all images with provenance
    images: List[Dict[str, Any]] = []
    for page in doc.pages:
        for image in page.images:
            images.append({
                "id": f"p{image.page_number}_i{image.image_index}",
                "page_number": image.page_number,
                "image_index": image.image_index,
                "filename": image.generate_filename(Path(doc.source_pdf).stem),
                "is_chart": image.is_likely_chart,
                "nearby_text": image.nearby_text,
                "dimensions": {
                    "width": image.width,
                    "height": image.height,
                },
                "bbox": list(image.bbox),
                "format": image.format,
            })

    # Collect section markers
    sections: List[Dict[str, Any]] = []
    for marker in doc.section_markers:
        sections.append({
            "page_number": marker.page_number,
            "section_type": marker.section_type,
            "title": marker.title,
            "heading_level": marker.heading_level,
            "bbox": list(marker.bbox),
        })

    # Calculate statistics
    total_text_blocks = sum(len(p.text_blocks) for p in doc.pages)
    total_tables = sum(len(p.tables) for p in doc.pages)
    total_images = sum(len(p.images) for p in doc.pages)

    return {
        "metadata": {
            "source_pdf": doc.source_pdf,
            "title": doc.title,
            "total_pages": doc.total_pages,
            "processed_at": datetime.now().isoformat(),
            "parser_version": PARSER_VERSION,
        },
        "statistics": {
            "text_blocks": total_text_blocks,
            "tables": total_tables,
            "images": total_images,
            "scanned_pages": doc.scanned_pages,
            "native_pages": doc.native_pages,
            "processing_time_seconds": doc.processing_time_seconds,
        },
        "sections": sections,
        "tables": tables,
        "images": images,
    }


def write_json(doc: ParsedDocument, output_path: Path) -> None:
    """Write JSON with ensure_ascii=False for Spanish characters.

    Args:
        doc: ParsedDocument to serialize.
        output_path: Path to write JSON file.
    """
    output = document_to_dict(doc)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
