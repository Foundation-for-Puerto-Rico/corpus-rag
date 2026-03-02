"""Text extraction module with structured TextBlock output.

Extracts text from PDF pages as structured TextBlock objects containing
text content, heading level, font metadata, and position information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF

from parser.extraction.heading_detector import HeadingMap, build_heading_map


@dataclass
class TextBlock:
    """Structured text block with formatting metadata.

    Attributes:
        text: The text content.
        heading_level: 0 for body text, 1-6 for h1-h6.
        is_bold: Whether the text is bold.
        is_italic: Whether the text is italic.
        font_size: Font size in points.
        bbox: Bounding box (x0, y0, x1, y1).
        page_number: Page number (1-indexed).
    """

    text: str
    heading_level: int
    is_bold: bool
    is_italic: bool
    font_size: float
    bbox: Tuple[float, float, float, float]
    page_number: int


def _round_to_half(size: float) -> float:
    """Round font size to nearest 0.5pt for heading map lookup."""
    return round(size * 2) / 2


def _is_bold(span: dict) -> bool:
    """Detect if a span is bold.

    Uses both font flags and font name heuristics since PDF font flags
    are often unreliable.

    Args:
        span: PyMuPDF span dictionary.

    Returns:
        True if text is bold.
    """
    # Check flags first (bit 4 = 16 is bold)
    flags = span.get("flags", 0)
    if flags & 16:
        return True

    # Fallback: check font name for "Bold"
    font_name = span.get("font", "")
    if "Bold" in font_name or "bold" in font_name:
        return True

    return False


def _is_italic(span: dict) -> bool:
    """Detect if a span is italic.

    Uses both font flags and font name heuristics.

    Args:
        span: PyMuPDF span dictionary.

    Returns:
        True if text is italic.
    """
    # Check flags first (bit 1 = 2 is italic)
    flags = span.get("flags", 0)
    if flags & 2:
        return True

    # Fallback: check font name for "Italic" or "Oblique"
    font_name = span.get("font", "")
    if "Italic" in font_name or "italic" in font_name:
        return True
    if "Oblique" in font_name or "oblique" in font_name:
        return True

    return False


def extract_text_blocks(
    page: fitz.Page,
    heading_map: HeadingMap,
    page_number: int,
) -> List[TextBlock]:
    """Extract structured text blocks from a page.

    Args:
        page: PyMuPDF Page object.
        heading_map: HeadingMap for heading level assignment.
        page_number: Page number (1-indexed).

    Returns:
        List of TextBlock objects in document order.
    """
    blocks: List[TextBlock] = []

    # Get text with full metadata
    page_dict = page.get_text("dict", flags=11)

    for block in page_dict.get("blocks", []):
        # Skip image blocks (they don't have "lines")
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = span.get("text", "")

                # Skip empty or whitespace-only spans
                if not text.strip():
                    continue

                font_size = span.get("size", 12.0)
                rounded_size = _round_to_half(font_size)
                heading_level = heading_map.get_level(rounded_size)

                bbox = span.get("bbox", (0, 0, 0, 0))

                text_block = TextBlock(
                    text=text,
                    heading_level=heading_level,
                    is_bold=_is_bold(span),
                    is_italic=_is_italic(span),
                    font_size=font_size,
                    bbox=tuple(bbox),  # Ensure it's a tuple
                    page_number=page_number,
                )
                blocks.append(text_block)

    return blocks


class TextExtractor:
    """Convenience class for extracting text from multi-page documents.

    Builds the heading map once and reuses it for all pages.
    """

    def __init__(self, doc: fitz.Document):
        """Initialize extractor with a PDF document.

        Args:
            doc: PyMuPDF Document object.
        """
        self.doc = doc
        self.heading_map = build_heading_map(doc)

    def extract_page(
        self,
        page: fitz.Page,
        page_number: int,
    ) -> List[TextBlock]:
        """Extract text blocks from a single page.

        Args:
            page: PyMuPDF Page object.
            page_number: Page number (1-indexed).

        Returns:
            List of TextBlock objects.
        """
        return extract_text_blocks(page, self.heading_map, page_number)

    def extract_all(self) -> List[TextBlock]:
        """Extract text blocks from all pages.

        Returns:
            List of TextBlock objects from all pages in document order.
        """
        all_blocks: List[TextBlock] = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = self.extract_page(page, page_num + 1)
            all_blocks.extend(blocks)
        return all_blocks
