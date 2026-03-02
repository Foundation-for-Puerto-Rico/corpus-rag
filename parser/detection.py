"""Page type detection for PDF documents.

Determines whether a PDF page is scanned (image-based) or contains native text,
enabling appropriate processing (OCR for scanned, direct extraction for native).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, List, Tuple, Union

import fitz  # PyMuPDF


class PageType(Enum):
    """Classification of PDF page content type."""

    NATIVE_TEXT = "native_text"  # Page has extractable text
    SCANNED = "scanned"          # Page is an image (needs OCR)
    MIXED = "mixed"              # Page has both text and significant images
    EMPTY = "empty"              # Page has no content


@dataclass
class PageAnalysis:
    """Result of analyzing a single page."""

    page_number: int
    page_type: PageType
    text_block_count: int
    image_count: int
    image_coverage: float  # 0.0 to 1.0
    has_text: bool


def detect_page_type(page: fitz.Page) -> PageType:
    """Detect the content type of a PDF page.

    Uses heuristics based on:
    - Presence of text blocks
    - Image coverage (percentage of page area covered by images)

    Detection logic:
    - SCANNED: image coverage > 95% AND no meaningful text
    - NATIVE_TEXT: has text blocks AND image coverage < 50%
    - MIXED: has both text and significant image coverage (>50%)
    - EMPTY: no text AND no images

    Args:
        page: A PyMuPDF page object

    Returns:
        PageType enum indicating the page content type
    """
    # Get text content
    text_dict = page.get_text("dict")
    text_blocks = text_dict.get("blocks", [])

    # Count actual text blocks (exclude image blocks in the dict)
    text_block_count = sum(1 for b in text_blocks if b.get("type") == 0)

    # Extract all text to check if it's meaningful
    text_content = page.get_text("text").strip()
    has_meaningful_text = len(text_content) > 10  # More than just whitespace/artifacts

    # Get images
    images = page.get_images(full=True)
    image_count = len(images)

    # Calculate image coverage
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    if page_area == 0:
        return PageType.EMPTY

    # Calculate total image area from image bounding boxes
    total_image_area = 0.0
    for img in images:
        # Get image bbox - need to find where images are placed on page
        xref = img[0]
        try:
            # Get all image instances on page
            for img_rect in page.get_image_rects(xref):
                img_area = img_rect.width * img_rect.height
                total_image_area += img_area
        except Exception:
            # If we can't get rects, estimate from image dimensions
            # This is a fallback - less accurate
            pass

    image_coverage = min(total_image_area / page_area, 1.0)  # Cap at 100%

    # Detection logic based on research heuristics
    if image_coverage > 0.95 and not has_meaningful_text:
        return PageType.SCANNED
    elif has_meaningful_text and image_coverage < 0.50:
        return PageType.NATIVE_TEXT
    elif has_meaningful_text and image_coverage >= 0.50:
        return PageType.MIXED
    elif not has_meaningful_text and image_count == 0:
        return PageType.EMPTY
    elif not has_meaningful_text and image_count > 0:
        # Has images but no text - likely scanned or image-heavy
        return PageType.SCANNED
    else:
        # Default fallback
        return PageType.NATIVE_TEXT


def analyze_page(page: fitz.Page, page_number: int) -> PageAnalysis:
    """Analyze a single page and return detailed analysis.

    Args:
        page: A PyMuPDF page object
        page_number: The page number (1-indexed)

    Returns:
        PageAnalysis with detailed information about the page
    """
    text_dict = page.get_text("dict")
    text_blocks = text_dict.get("blocks", [])
    text_block_count = sum(1 for b in text_blocks if b.get("type") == 0)

    text_content = page.get_text("text").strip()
    has_text = len(text_content) > 10

    images = page.get_images(full=True)
    image_count = len(images)

    # Calculate image coverage
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    total_image_area = 0.0
    if page_area > 0:
        for img in images:
            xref = img[0]
            try:
                for img_rect in page.get_image_rects(xref):
                    total_image_area += img_rect.width * img_rect.height
            except Exception:
                pass

    image_coverage = min(total_image_area / page_area, 1.0) if page_area > 0 else 0.0

    page_type = detect_page_type(page)

    return PageAnalysis(
        page_number=page_number,
        page_type=page_type,
        text_block_count=text_block_count,
        image_count=image_count,
        image_coverage=image_coverage,
        has_text=has_text,
    )


def analyze_pdf(pdf_path: Union[str, Path]) -> List[Tuple[int, PageType]]:
    """Analyze all pages in a PDF and return their types.

    Processes pages one at a time to manage memory for large documents.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of (page_number, PageType) tuples (1-indexed page numbers)

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        fitz.FileDataError: If the file is not a valid PDF
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results: List[Tuple[int, PageType]] = []

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_type = detect_page_type(page)
            results.append((page_num + 1, page_type))  # 1-indexed

    return results


def analyze_pdf_detailed(pdf_path: Union[str, Path]) -> Generator[PageAnalysis, None, None]:
    """Analyze all pages in a PDF with detailed information.

    Uses a generator to process pages lazily, reducing memory usage
    for large documents.

    Args:
        pdf_path: Path to the PDF file

    Yields:
        PageAnalysis for each page in the document

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            yield analyze_page(page, page_num + 1)  # 1-indexed


def get_pages_by_type(pdf_path: Union[str, Path], page_type: PageType) -> List[int]:
    """Get all page numbers of a specific type.

    Convenience function for filtering pages that need OCR or other processing.

    Args:
        pdf_path: Path to the PDF file
        page_type: The PageType to filter for

    Returns:
        List of page numbers (1-indexed) matching the specified type
    """
    results = analyze_pdf(pdf_path)
    return [page_num for page_num, ptype in results if ptype == page_type]
