"""Image extraction module for charts, graphs, and visual content."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Generator

import fitz

from parser.models.image import ExtractedImage


# Minimum dimensions to filter out icons/bullets (research: 100px)
MIN_IMAGE_SIZE = 100


def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_number: int,
    min_size: int = MIN_IMAGE_SIZE,
) -> list[ExtractedImage]:
    """Extract images from a single page.

    Args:
        doc: PyMuPDF Document (needed for extract_image)
        page: PyMuPDF Page object
        page_number: 1-indexed page number
        min_size: Minimum dimension to consider (filters icons)

    Returns:
        List of ExtractedImage objects (filtered by size)
    """
    images = []
    image_list = page.get_images(full=True)
    image_info = page.get_image_info()

    for i, img in enumerate(image_list):
        xref = img[0]
        base = doc.extract_image(xref)

        # Filter small images (icons, bullets)
        if base["width"] < min_size or base["height"] < min_size:
            continue

        # Get bounding box (may not match 1:1 with image_list)
        bbox = (0.0, 0.0, 0.0, 0.0)
        if i < len(image_info):
            bbox = image_info[i].get("bbox", bbox)
            bbox = tuple(bbox) if bbox else (0.0, 0.0, 0.0, 0.0)

        # Get nearby text for potential caption
        nearby_text = ""
        if bbox != (0.0, 0.0, 0.0, 0.0):
            rect = fitz.Rect(bbox)
            # Expand search area: slightly left/right, more below (captions usually below)
            expanded = rect + (-20, -20, 20, 60)
            nearby_text = page.get_text("text", clip=expanded)[:200].strip()

        images.append(ExtractedImage(
            page_number=page_number,
            image_index=i,
            xref=xref,
            bbox=bbox,
            width=base["width"],
            height=base["height"],
            format=base["ext"],
            image_data=base["image"],
            nearby_text=nearby_text,
        ))

    return images


class ImageExtractor:
    """Extract images from multi-page PDF documents.

    Follows same pattern as TextExtractor and TableExtractor.
    Uses generator pattern for memory efficiency (images are large).
    """

    def __init__(self, doc: fitz.Document):
        """Initialize with PDF document.

        Args:
            doc: PyMuPDF Document object
        """
        self.doc = doc

    def extract_page(self, page_number: int, min_size: int = MIN_IMAGE_SIZE) -> list[ExtractedImage]:
        """Extract images from a single page.

        Args:
            page_number: 1-indexed page number
            min_size: Minimum dimension to consider

        Returns:
            List of ExtractedImage objects
        """
        page = self.doc[page_number - 1]
        return extract_images_from_page(self.doc, page, page_number, min_size)

    def extract_all(self, min_size: int = MIN_IMAGE_SIZE) -> Generator[ExtractedImage, None, None]:
        """Extract images from all pages as generator.

        Memory efficient: yields one page's images at a time,
        calls gc.collect() between pages (per Phase 1 pattern).

        Args:
            min_size: Minimum dimension to consider

        Yields:
            ExtractedImage objects from all pages
        """
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            images = extract_images_from_page(self.doc, page, page_num + 1, min_size)
            yield from images
            gc.collect()  # Memory safety for large PDFs (DEC-01-01-03)

    def extract_charts(self) -> Generator[ExtractedImage, None, None]:
        """Extract only likely chart/graph images.

        Uses is_likely_chart heuristic to filter.

        Yields:
            ExtractedImage objects that appear to be charts
        """
        for img in self.extract_all():
            if img.is_likely_chart:
                yield img

    def save_images(
        self,
        output_dir: str | Path,
        pdf_name: str,
        min_size: int = MIN_IMAGE_SIZE,
        charts_only: bool = False,
    ) -> list[str]:
        """Extract and save images to disk.

        Args:
            output_dir: Directory to save images
            pdf_name: Base name for filenames
            min_size: Minimum dimension to consider
            charts_only: If True, only save likely charts

        Returns:
            List of saved filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved = []
        source = self.extract_charts() if charts_only else self.extract_all(min_size)

        for img in source:
            filename = img.generate_filename(pdf_name)
            filepath = output_path / filename

            with open(filepath, "wb") as f:
                f.write(img.image_data)

            saved.append(filename)

        return saved

    def get_image_count(self, min_size: int = MIN_IMAGE_SIZE) -> int:
        """Get total image count (filtered by size)."""
        return sum(1 for _ in self.extract_all(min_size))

    def get_chart_count(self) -> int:
        """Get count of likely charts."""
        return sum(1 for _ in self.extract_charts())
