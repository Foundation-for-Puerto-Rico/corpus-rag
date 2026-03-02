"""Unified PDF processor that integrates detection and OCR.

Routes pages based on their detected type: native text pages are extracted directly,
scanned pages go through OCR, mixed pages get both treatments.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import fitz  # PyMuPDF

from parser.detection import PageType, detect_page_type
from parser.ocr import OCRResult, ocr_page


@dataclass
class ProcessedPage:
    """Result of processing a single PDF page.

    Attributes:
        page_number: Page number in the document (1-indexed)
        page_type: Detected page type (NATIVE_TEXT, SCANNED, MIXED, EMPTY)
        text: Extracted text content
        ocr_applied: Whether OCR was used for text extraction
        confidence: OCR confidence score (0.0-1.0), None if OCR not applied
    """

    page_number: int
    page_type: PageType
    text: str
    ocr_applied: bool
    confidence: Optional[float]


class PDFProcessor:
    """Unified processor for PDF documents.

    Automatically detects page types and applies appropriate extraction:
    - NATIVE_TEXT pages: Direct text extraction
    - SCANNED pages: OCR processing
    - MIXED pages: Direct extraction (text layer is accessible)
    - EMPTY pages: Return empty text

    Designed for memory efficiency with generator-based streaming.
    """

    def __init__(self, language: str = "spa"):
        """Initialize the processor.

        Args:
            language: Tesseract language code for OCR (default: 'spa' for Spanish)
        """
        self.language = language

    def process_page(self, page: fitz.Page, page_num: int) -> ProcessedPage:
        """Process a single PDF page.

        Detects the page type and extracts text accordingly.

        Args:
            page: A PyMuPDF page object
            page_num: Page number (1-indexed)

        Returns:
            ProcessedPage with extracted text and metadata
        """
        page_type = detect_page_type(page)

        text = ""
        ocr_applied = False
        confidence: Optional[float] = None

        if page_type == PageType.EMPTY:
            # Empty page - no text to extract
            text = ""
            ocr_applied = False

        elif page_type == PageType.SCANNED:
            # Scanned page - need OCR
            ocr_result: OCRResult = ocr_page(
                page,
                language=self.language,
                page_number=page_num,
            )
            text = ocr_result.text
            confidence = ocr_result.confidence
            ocr_applied = True

        elif page_type in (PageType.NATIVE_TEXT, PageType.MIXED):
            # Native text or mixed - extract text directly
            text = page.get_text("text")
            ocr_applied = False

        return ProcessedPage(
            page_number=page_num,
            page_type=page_type,
            text=text,
            ocr_applied=ocr_applied,
            confidence=confidence,
        )

    def process_pdf(
        self, pdf_path: Union[str, Path]
    ) -> Generator[ProcessedPage, None, None]:
        """Process all pages in a PDF document.

        Yields processed pages one at a time for memory efficiency.
        Calls gc.collect() after each page to prevent memory accumulation.

        Args:
            pdf_path: Path to the PDF file

        Yields:
            ProcessedPage for each page in the document

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            fitz.FileDataError: If the file is not a valid PDF
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = None
        try:
            doc = fitz.open(pdf_path)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1  # Convert to 1-indexed

                processed = self.process_page(page, page_num)
                yield processed

                # Memory cleanup after each page
                gc.collect()

        finally:
            if doc is not None:
                doc.close()
