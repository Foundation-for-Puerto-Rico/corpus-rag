"""OCR processing for scanned PDF pages.

Extracts text from scanned pages using Tesseract OCR with Spanish language support.
Handles UTF-8 encoding to preserve Spanish characters (accents, n with tilde, etc.).
"""

from __future__ import annotations

import gc
import io
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


class TesseractNotFoundError(Exception):
    """Raised when Tesseract OCR is not installed or not found in PATH."""

    pass


class LanguageNotAvailableError(Exception):
    """Raised when the requested language pack is not installed for Tesseract."""

    pass


@dataclass
class OCRResult:
    """Result of OCR processing on a single page.

    Attributes:
        text: Extracted text content (UTF-8 encoded)
        confidence: OCR confidence score (0.0-1.0), if available
        language: Language code used for OCR (e.g., 'spa' for Spanish)
        page_number: Page number in the source document (1-indexed)
    """

    text: str
    confidence: Optional[float]
    language: str
    page_number: int


def check_tesseract_installed() -> bool:
    """Check if Tesseract OCR is installed and accessible.

    Returns:
        True if Tesseract is installed and accessible

    Raises:
        TesseractNotFoundError: If Tesseract is not found
    """
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        raise TesseractNotFoundError(
            "Tesseract OCR is not installed. "
            "Install it with: brew install tesseract tesseract-lang (macOS) "
            "or apt-get install tesseract-ocr tesseract-ocr-spa (Ubuntu)"
        )
    except subprocess.TimeoutExpired:
        raise TesseractNotFoundError("Tesseract command timed out")

    raise TesseractNotFoundError("Tesseract OCR returned an error")


def check_language_available(language: str) -> bool:
    """Check if a specific language pack is installed for Tesseract.

    Args:
        language: Tesseract language code (e.g., 'spa' for Spanish)

    Returns:
        True if the language is available

    Raises:
        LanguageNotAvailableError: If the language pack is not installed
    """
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        available_langs = result.stdout.strip().split("\n")[1:]  # Skip header line
        available_langs = [lang.strip() for lang in available_langs]

        if language in available_langs:
            return True
        else:
            raise LanguageNotAvailableError(
                f"Language '{language}' is not installed for Tesseract. "
                f"Available languages: {', '.join(available_langs[:10])}... "
                f"Install with: brew install tesseract-lang (macOS) "
                f"or apt-get install tesseract-ocr-{language} (Ubuntu)"
            )
    except FileNotFoundError:
        raise TesseractNotFoundError("Tesseract OCR is not installed")


def _validate_spanish_encoding(text: str) -> bool:
    """Check if Spanish text appears to be properly encoded.

    Looks for common issues like replacement characters or mojibake.

    Args:
        text: Text to validate

    Returns:
        True if encoding appears valid, False if issues detected
    """
    # Common Spanish characters that should be present in Spanish text
    spanish_chars = set("aeiouaeiounAEIOUAEIOUN")

    # Characters that indicate encoding problems
    problem_chars = {
        "\ufffd",  # Unicode replacement character
        "?",  # May indicate encoding issues in some contexts
    }

    # Count replacement characters
    replacement_count = sum(1 for c in text if c == "\ufffd")

    # If more than 1% are replacement characters, likely encoding issue
    if len(text) > 0 and replacement_count / len(text) > 0.01:
        logger.warning(
            f"Possible encoding issue: {replacement_count} replacement characters "
            f"in text of length {len(text)}"
        )
        return False

    return True


def ocr_page(
    page: fitz.Page,
    language: str = "spa",
    dpi: int = 300,
    page_number: int = 0,
) -> OCRResult:
    """Extract text from a single PDF page using OCR.

    Converts the page to an image at the specified DPI, then uses Tesseract
    OCR to extract text. Optimized for Spanish language documents.

    Args:
        page: A PyMuPDF page object
        language: Tesseract language code (default: 'spa' for Spanish)
        dpi: Resolution for image conversion (default: 300 for quality)
        page_number: Page number for the result (1-indexed)

    Returns:
        OCRResult containing extracted text and metadata

    Raises:
        TesseractNotFoundError: If Tesseract is not installed
        LanguageNotAvailableError: If the language pack is not installed
    """
    # Import pytesseract here to catch import errors gracefully
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract is not installed. Install with: pip install pytesseract"
        )

    # Convert page to image at specified DPI
    # Using matrix for proper DPI scaling
    zoom = dpi / 72  # Default PDF resolution is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Convert pixmap to PIL Image for pytesseract
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))

    # Perform OCR
    try:
        # Get text with confidence data
        text = pytesseract.image_to_string(
            image,
            lang=language,
            config="--psm 1",  # Automatic page segmentation with OSD
        )

        # Get confidence data (optional, may not always be available)
        try:
            data = pytesseract.image_to_data(
                image,
                lang=language,
                output_type=pytesseract.Output.DICT,
            )
            # Calculate average confidence from word confidences
            confidences = [
                int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) > 0
            ]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else None
        except Exception:
            avg_confidence = None

    except pytesseract.TesseractNotFoundError:
        raise TesseractNotFoundError(
            "Tesseract OCR is not installed or not in PATH. "
            "Install with: brew install tesseract tesseract-lang (macOS) "
            "or apt-get install tesseract-ocr tesseract-ocr-spa (Ubuntu)"
        )

    # Ensure UTF-8 encoding
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Validate encoding
    if not _validate_spanish_encoding(text):
        logger.warning(f"Page {page_number}: Spanish encoding may be corrupted")

    # Clean up
    del pix
    del image
    gc.collect()

    return OCRResult(
        text=text,
        confidence=avg_confidence,
        language=language,
        page_number=page_number,
    )


def ocr_pdf_pages(
    pdf_path: Union[str, Path],
    page_numbers: List[int],
    language: str = "spa",
    dpi: int = 300,
) -> List[OCRResult]:
    """Process multiple pages of a PDF with OCR.

    Processes pages one at a time to manage memory for large documents.
    Calls gc.collect() after each page to free memory.

    Args:
        pdf_path: Path to the PDF file
        page_numbers: List of page numbers to process (1-indexed)
        language: Tesseract language code (default: 'spa' for Spanish)
        dpi: Resolution for image conversion (default: 300)

    Returns:
        List of OCRResult objects for each processed page

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        TesseractNotFoundError: If Tesseract is not installed
        LanguageNotAvailableError: If the language pack is not installed
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Verify Tesseract and language before processing
    check_tesseract_installed()
    check_language_available(language)

    results: List[OCRResult] = []

    with fitz.open(pdf_path) as doc:
        for page_num in page_numbers:
            # Convert to 0-indexed for PyMuPDF
            page_idx = page_num - 1

            if page_idx < 0 or page_idx >= len(doc):
                logger.warning(f"Page {page_num} out of range, skipping")
                continue

            page = doc[page_idx]

            try:
                result = ocr_page(page, language=language, dpi=dpi, page_number=page_num)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                # Create empty result for failed page
                results.append(
                    OCRResult(
                        text="",
                        confidence=None,
                        language=language,
                        page_number=page_num,
                    )
                )

            # Memory cleanup after each page
            gc.collect()

    return results


def get_tesseract_version() -> Optional[str]:
    """Get the installed Tesseract version.

    Returns:
        Version string or None if not installed
    """
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # First line contains version info
            return result.stdout.split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_available_languages() -> List[str]:
    """Get list of available Tesseract language packs.

    Returns:
        List of language codes, or empty list if Tesseract not installed
    """
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            langs = result.stdout.strip().split("\n")[1:]  # Skip header
            return [lang.strip() for lang in langs if lang.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []
