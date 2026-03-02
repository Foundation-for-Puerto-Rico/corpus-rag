"""Section marker module for identifying document sections.

Provides detection of Spanish appendix sections (anejo, anexo, apendice)
and other document structure markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from parser.extraction.text_extractor import TextBlock


# Patterns for Spanish appendix identification
# Handle both with and without accent variations
ANEJO_PATTERNS = [
    r"\b(ANEJO|ANEJOS)\b",
    r"\b(ANEXO|ANEXOS)\b",
    r"\b(AP[EÉ]NDICE|AP[EÉ]NDICES)\b",  # Handle APENDICE and APENDICE
]

# Compiled pattern for efficient matching
SECTION_PATTERN = re.compile(
    "|".join(ANEJO_PATTERNS),
    re.IGNORECASE,
)


@dataclass
class SectionMarker:
    """Marks a document section.

    Attributes:
        page_number: Page where the section marker appears (1-indexed).
        section_type: Type of section ("main", "anejo", "anexo", "apendice").
        title: Full text of the section heading.
        heading_level: Markdown heading level (1-6).
        bbox: Bounding box of the section marker (x0, y0, x1, y1).
    """

    page_number: int
    section_type: str
    title: str
    heading_level: int
    bbox: Tuple[float, float, float, float]


def get_section_type(text: str) -> str:
    """Determine the section type from text content.

    Args:
        text: Text to analyze.

    Returns:
        Section type: "anejo", "anexo", "apendice", or "main".
    """
    text_upper = text.upper()

    # Check in order of specificity
    if re.search(r"\bANEJO", text_upper):
        return "anejo"
    if re.search(r"\bANEXO", text_upper):
        return "anexo"
    if re.search(r"\bAP[EÉ]NDICE", text_upper):
        return "apendice"

    return "main"


def is_appendix_marker(text: str, heading_level: int) -> bool:
    """Check if text is an appendix section marker.

    An appendix marker must:
    1. Match one of the appendix patterns (anejo, anexo, apendice)
    2. Be a heading (heading_level > 0), not body text

    This prevents inline mentions like "ver anejo A" from being
    detected as section markers.

    Args:
        text: Text to check.
        heading_level: Markdown heading level (0 for body, 1-6 for headings).

    Returns:
        True if text is an appendix section marker.
    """
    # Must be a heading, not body text
    if heading_level == 0:
        return False

    # Must match appendix pattern
    return bool(SECTION_PATTERN.search(text))


def detect_appendix_sections(blocks: List["TextBlock"]) -> List[SectionMarker]:
    """Find all appendix/anejo sections in a list of TextBlocks.

    Scans through blocks looking for headings that match appendix patterns.
    Body text mentions (heading_level=0) are ignored.

    Args:
        blocks: List of TextBlock objects to scan.

    Returns:
        List of SectionMarker objects for detected appendix sections.
    """
    markers: List[SectionMarker] = []

    for block in blocks:
        if is_appendix_marker(block.text, block.heading_level):
            section_type = get_section_type(block.text)
            marker = SectionMarker(
                page_number=block.page_number,
                section_type=section_type,
                title=block.text.strip(),
                heading_level=block.heading_level,
                bbox=block.bbox,
            )
            markers.append(marker)

    return markers


def detect_all_sections(blocks: List["TextBlock"]) -> List[SectionMarker]:
    """Find all section headings in a list of TextBlocks.

    Similar to detect_appendix_sections but also includes main document
    sections (any heading, not just appendix markers).

    Args:
        blocks: List of TextBlock objects to scan.

    Returns:
        List of SectionMarker objects for all sections.
    """
    markers: List[SectionMarker] = []

    for block in blocks:
        if block.heading_level > 0:
            # Determine if it's an appendix or main section
            if is_appendix_marker(block.text, block.heading_level):
                section_type = get_section_type(block.text)
            else:
                section_type = "main"

            marker = SectionMarker(
                page_number=block.page_number,
                section_type=section_type,
                title=block.text.strip(),
                heading_level=block.heading_level,
                bbox=block.bbox,
            )
            markers.append(marker)

    return markers
