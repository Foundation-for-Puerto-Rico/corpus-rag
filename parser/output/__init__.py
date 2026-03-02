"""Output module for converting extracted text to various formats.

Provides Markdown conversion with heading levels and formatting,
document aggregation with ParsedDocument/ParsedPage dataclasses,
and JSON output with provenance metadata.
"""

from parser.output.markdown import (
    MarkdownWriter,
    blocks_to_markdown,
    page_to_markdown,
)
from parser.output.document import (
    ParsedDocument,
    ParsedPage,
    DocumentWriter,
    interleave_content,
    filter_text_in_tables,
)
from parser.output.json_writer import (
    document_to_dict,
    write_json,
)

__all__ = [
    "MarkdownWriter",
    "blocks_to_markdown",
    "page_to_markdown",
    "ParsedDocument",
    "ParsedPage",
    "DocumentWriter",
    "interleave_content",
    "filter_text_in_tables",
    "document_to_dict",
    "write_json",
]
