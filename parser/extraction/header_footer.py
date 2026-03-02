"""Header/footer detection and filtering module.

Provides frequency-based detection of repetitive headers and footers
that appear across multiple pages in PDF documents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Set, Tuple

import fitz  # PyMuPDF

if TYPE_CHECKING:
    from parser.extraction.text_extractor import TextBlock


# Zone defaults (in points, 72 points = 1 inch)
DEFAULT_HEADER_ZONE = 72.0  # 1 inch from top
DEFAULT_FOOTER_ZONE = 72.0  # 1 inch from bottom
MIN_OCCURRENCES = 3  # Must appear on at least 3 pages


def _normalize_text(text: str, max_chars: int = 100) -> str:
    """Normalize text for comparison.

    Strips whitespace and truncates to first max_chars characters
    to handle minor variations in repetitive text.

    Args:
        text: Text to normalize.
        max_chars: Maximum characters to keep.

    Returns:
        Normalized text string.
    """
    return text.strip()[:max_chars].strip()


def detect_repetitive_headers_footers(
    doc: fitz.Document,
    header_zone: float = DEFAULT_HEADER_ZONE,
    footer_zone: float = DEFAULT_FOOTER_ZONE,
    min_occurrences: int = MIN_OCCURRENCES,
) -> Tuple[Set[str], Set[str]]:
    """Detect text that repeats across pages in header/footer zones.

    Analyzes all pages to find text blocks that appear in header or footer
    zones on multiple pages. Text must appear on at least min_occurrences
    pages to be considered repetitive (to avoid false positives on
    legitimate content that happens to be near page edges).

    Args:
        doc: PyMuPDF Document object.
        header_zone: Distance from top of page (in points) defining header zone.
        footer_zone: Distance from bottom of page (in points) defining footer zone.
        min_occurrences: Minimum number of pages text must appear on.

    Returns:
        Tuple of (headers_to_filter, footers_to_filter) - sets of normalized
        text strings that should be filtered.
    """
    header_texts: Counter = Counter()
    footer_texts: Counter = Counter()

    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("dict", flags=11).get("blocks", [])

        for block in blocks:
            if "lines" not in block:
                continue

            bbox = block.get("bbox", (0, 0, 0, 0))
            y_top = bbox[1]
            y_bottom = bbox[3]

            # Collect all text in block
            text_parts = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span.get("text", ""))

            full_text = "".join(text_parts)
            if not full_text.strip():
                continue

            # Normalize for comparison
            normalized = _normalize_text(full_text)
            if not normalized:
                continue

            # Check if in header zone (near top)
            if y_top < header_zone:
                header_texts[normalized] += 1
            # Check if in footer zone (near bottom)
            elif y_bottom > page_height - footer_zone:
                footer_texts[normalized] += 1

    # Return only texts that appear on multiple pages
    headers_to_filter = {
        text for text, count in header_texts.items() if count >= min_occurrences
    }
    footers_to_filter = {
        text for text, count in footer_texts.items() if count >= min_occurrences
    }

    return headers_to_filter, footers_to_filter


@dataclass
class HeaderFooterFilter:
    """Filter for identifying repetitive headers/footers in TextBlocks.

    Attributes:
        headers_to_skip: Set of normalized header text strings to filter.
        footers_to_skip: Set of normalized footer text strings to filter.
        header_zone: Distance from top of page defining header zone.
        footer_zone: Distance from bottom of page defining footer zone.
    """

    headers_to_skip: Set[str]
    footers_to_skip: Set[str]
    header_zone: float = DEFAULT_HEADER_ZONE
    footer_zone: float = DEFAULT_FOOTER_ZONE

    def should_skip(self, block: "TextBlock", page_height: float) -> bool:
        """Check if a TextBlock should be filtered as repetitive header/footer.

        A block is filtered if:
        1. It is in a header/footer zone (based on bbox position), AND
        2. Its normalized text matches a known repetitive header/footer

        Args:
            block: TextBlock to check.
            page_height: Height of the page in points.

        Returns:
            True if block should be filtered, False otherwise.
        """
        y_top = block.bbox[1]
        y_bottom = block.bbox[3]
        normalized_text = _normalize_text(block.text)

        # Check header zone
        if y_top < self.header_zone:
            if normalized_text in self.headers_to_skip:
                return True

        # Check footer zone
        if y_bottom > page_height - self.footer_zone:
            if normalized_text in self.footers_to_skip:
                return True

        return False


def build_filter(
    doc: fitz.Document,
    header_zone: float = DEFAULT_HEADER_ZONE,
    footer_zone: float = DEFAULT_FOOTER_ZONE,
    min_occurrences: int = MIN_OCCURRENCES,
) -> HeaderFooterFilter:
    """Build a HeaderFooterFilter from a document.

    Convenience function that detects repetitive headers/footers
    and creates a filter for them.

    Args:
        doc: PyMuPDF Document object.
        header_zone: Distance from top of page defining header zone.
        footer_zone: Distance from bottom of page defining footer zone.
        min_occurrences: Minimum occurrences to be considered repetitive.

    Returns:
        HeaderFooterFilter instance configured for this document.
    """
    headers, footers = detect_repetitive_headers_footers(
        doc, header_zone, footer_zone, min_occurrences
    )

    return HeaderFooterFilter(
        headers_to_skip=headers,
        footers_to_skip=footers,
        header_zone=header_zone,
        footer_zone=footer_zone,
    )
