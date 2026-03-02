"""Extraction module for structured text extraction from PDFs.

Provides heading detection, text block extraction with font metadata,
header/footer filtering, section marker detection, table extraction,
and image extraction.
"""

from parser.extraction.heading_detector import HeadingMap, build_heading_map
from parser.extraction.text_extractor import TextBlock, TextExtractor, extract_text_blocks
from parser.extraction.table_extractor import (
    TableExtractor,
    extract_tables_from_page,
    fill_merged_cells,
    _is_valid_borderless_table,
    MIN_BORDERLESS_ROWS,
    MIN_BORDERLESS_COLS,
    MIN_WORDS_VERTICAL,
    MIN_WORDS_HORIZONTAL,
)
from parser.extraction.image_extractor import (
    ImageExtractor,
    extract_images_from_page,
    MIN_IMAGE_SIZE,
)
from parser.extraction.header_footer import (
    HeaderFooterFilter,
    build_filter,
    detect_repetitive_headers_footers,
    DEFAULT_HEADER_ZONE,
    DEFAULT_FOOTER_ZONE,
    MIN_OCCURRENCES,
)
from parser.extraction.section_marker import (
    SectionMarker,
    ANEJO_PATTERNS,
    SECTION_PATTERN,
    is_appendix_marker,
    detect_appendix_sections,
    detect_all_sections,
    get_section_type,
)
from parser.models.table import ExtractedTable

__all__ = [
    # Heading detection
    "HeadingMap",
    "build_heading_map",
    # Text extraction
    "TextBlock",
    "TextExtractor",
    "extract_text_blocks",
    # Table extraction
    "TableExtractor",
    "extract_tables_from_page",
    "fill_merged_cells",
    "_is_valid_borderless_table",
    "MIN_BORDERLESS_ROWS",
    "MIN_BORDERLESS_COLS",
    "MIN_WORDS_VERTICAL",
    "MIN_WORDS_HORIZONTAL",
    "ExtractedTable",
    # Image extraction
    "ImageExtractor",
    "extract_images_from_page",
    "MIN_IMAGE_SIZE",
    # Header/footer filtering
    "HeaderFooterFilter",
    "build_filter",
    "detect_repetitive_headers_footers",
    "DEFAULT_HEADER_ZONE",
    "DEFAULT_FOOTER_ZONE",
    "MIN_OCCURRENCES",
    # Section markers
    "SectionMarker",
    "ANEJO_PATTERNS",
    "SECTION_PATTERN",
    "is_appendix_marker",
    "detect_appendix_sections",
    "detect_all_sections",
    "get_section_type",
]
