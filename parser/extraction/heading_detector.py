"""Heading detection module for font-based heading level assignment.

Analyzes PDF documents to build a mapping of font sizes to Markdown heading levels.
Uses frequency analysis to identify body text (most common size) and assigns
heading levels to larger font sizes in descending order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict

import fitz  # PyMuPDF


@dataclass
class HeadingMap:
    """Maps font sizes to heading levels.

    Attributes:
        size_to_level: Dict mapping rounded font size to heading level.
                       Level 0 = body text, levels 1-6 = h1-h6.
        body_size: The most common font size (body text).
        heading_sizes: List of font sizes larger than body, sorted descending.
    """

    size_to_level: Dict[float, int] = field(default_factory=dict)
    body_size: float = 12.0
    heading_sizes: list = field(default_factory=list)

    def get_level(self, font_size: float) -> int:
        """Get heading level for a font size.

        Args:
            font_size: The font size to look up.

        Returns:
            Heading level (0 for body, 1-6 for headings).
            Returns 0 if size not in map.
        """
        # Round to nearest 0.5pt for lookup
        rounded = round(font_size * 2) / 2
        return self.size_to_level.get(rounded, 0)


def _round_to_half(size: float) -> float:
    """Round font size to nearest 0.5pt.

    This handles font size rounding errors where the same visual font
    may appear as 11.9pt on one page and 12.1pt on another.

    Args:
        size: Font size in points.

    Returns:
        Font size rounded to nearest 0.5pt.
    """
    return round(size * 2) / 2


def build_heading_map(doc: fitz.Document) -> HeadingMap:
    """Build mapping of font sizes to heading levels.

    Analyzes entire document to find font size distribution,
    identifies body text as most common size, then assigns
    heading levels to larger fonts.

    Args:
        doc: PyMuPDF Document object.

    Returns:
        HeadingMap with size-to-level mapping.
    """
    font_sizes: Counter = Counter()

    # Collect font sizes from all pages
    for page in doc:
        blocks = page.get_text("dict", flags=11)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = _round_to_half(span["size"])
                    # Count by character count for better weighting
                    text_len = len(span["text"].strip())
                    if text_len > 0:
                        font_sizes[size] += text_len

    # Handle edge case: no text found
    if not font_sizes:
        return HeadingMap(
            size_to_level={12.0: 0},
            body_size=12.0,
            heading_sizes=[],
        )

    # Body text is the most common size
    body_size = font_sizes.most_common(1)[0][0]

    # Sizes larger than body are potential headings, sorted descending
    heading_sizes = sorted(
        [s for s in font_sizes if s > body_size],
        reverse=True,
    )

    # Build the mapping
    size_to_level: Dict[float, int] = {}

    # Body text and anything smaller is level 0
    for size in font_sizes:
        if size <= body_size:
            size_to_level[size] = 0

    # Heading sizes get levels 1-6 (h1-h6), cap at 6
    for i, size in enumerate(heading_sizes[:6]):
        size_to_level[size] = i + 1

    # Any remaining heading sizes (7+) get capped at h6
    for size in heading_sizes[6:]:
        size_to_level[size] = 6

    return HeadingMap(
        size_to_level=size_to_level,
        body_size=body_size,
        heading_sizes=heading_sizes,
    )
