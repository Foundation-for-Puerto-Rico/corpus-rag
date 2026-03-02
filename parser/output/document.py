"""Document output module for aggregating and writing parsed PDF content.

Provides ParsedDocument and ParsedPage dataclasses for content aggregation,
interleaving functions for Y-position ordering, and DocumentWriter for
generating Markdown output files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from parser.extraction.text_extractor import TextBlock
from parser.extraction.section_marker import SectionMarker
from parser.models.table import ExtractedTable
from parser.models.image import ExtractedImage
from parser.output.markdown import blocks_to_markdown

if TYPE_CHECKING:
    from parser.extraction.header_footer import HeaderFooterFilter


@dataclass
class ParsedPage:
    """Aggregated content from a single PDF page.

    Attributes:
        page_number: Page number (1-indexed).
        text_blocks: Text blocks extracted from the page.
        tables: Tables extracted from the page.
        images: Images extracted from the page.
    """
    page_number: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Complete parsed PDF document with all extracted content.

    Attributes:
        source_pdf: Original PDF filename.
        title: Document title (first h1 or filename).
        total_pages: Total number of pages in the PDF.
        pages: List of ParsedPage objects.
        section_markers: Detected section markers (anejos, etc.).
        scanned_pages: Count of pages processed via OCR.
        native_pages: Count of pages with native text.
        processing_time_seconds: Time taken to process the document.
    """
    source_pdf: str
    title: str
    total_pages: int
    pages: List[ParsedPage] = field(default_factory=list)
    section_markers: List[SectionMarker] = field(default_factory=list)
    scanned_pages: int = 0
    native_pages: int = 0
    processing_time_seconds: float = 0.0


def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box.

    Args:
        bbox: Bounding box (x0, y0, x1, y1).

    Returns:
        Center point (cx, cy).
    """
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _point_in_bbox(
    point: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    padding: float = 0.0,
) -> bool:
    """Check if a point is inside a bounding box with optional padding.

    Args:
        point: Point (x, y) to check.
        bbox: Bounding box (x0, y0, x1, y1).
        padding: Extra padding around the bbox.

    Returns:
        True if point is inside the padded bbox.
    """
    x, y = point
    x0, y0, x1, y1 = bbox
    return (
        x0 - padding <= x <= x1 + padding and
        y0 - padding <= y <= y1 + padding
    )


def filter_text_in_tables(
    text_blocks: List[TextBlock],
    tables: List[ExtractedTable],
    padding: float = 5.0,
) -> List[TextBlock]:
    """Filter out text blocks that fall inside table bounding boxes.

    This prevents duplicate content from appearing in the output, as table
    text is already captured in the table structure.

    Args:
        text_blocks: List of TextBlock objects.
        tables: List of ExtractedTable objects.
        padding: Padding around table bbox (in points).

    Returns:
        List of TextBlock objects that are not inside any table.
    """
    if not tables:
        return text_blocks

    filtered: List[TextBlock] = []

    for block in text_blocks:
        block_center = _bbox_center(block.bbox)
        inside_table = False

        for table in tables:
            if _point_in_bbox(block_center, table.bbox, padding):
                inside_table = True
                break

        if not inside_table:
            filtered.append(block)

    return filtered


def interleave_content(
    text_blocks: List[TextBlock],
    tables: List[ExtractedTable],
    images: List[ExtractedImage],
    table_padding: float = 5.0,
) -> List[Tuple[str, float, Any]]:
    """Order content by Y position, filtering text inside table bboxes.

    Creates a unified ordering of all content types by their vertical position
    on the page, ensuring text that appears inside table bounding boxes is
    excluded to prevent duplication.

    Args:
        text_blocks: List of TextBlock objects.
        tables: List of ExtractedTable objects.
        images: List of ExtractedImage objects.
        table_padding: Padding for table-text collision detection.

    Returns:
        List of (content_type, y_pos, content) tuples sorted by Y position.
        content_type is one of: "text", "table", "image"
    """
    # Filter out text blocks inside tables
    filtered_text = filter_text_in_tables(text_blocks, tables, table_padding)

    items: List[Tuple[str, float, Any]] = []

    # Add text blocks
    for block in filtered_text:
        y_pos = block.bbox[1]  # y0
        items.append(("text", y_pos, block))

    # Add tables
    for table in tables:
        y_pos = table.bbox[1]  # y0
        items.append(("table", y_pos, table))

    # Add images
    for image in images:
        y_pos = image.bbox[1]  # y0
        items.append(("image", y_pos, image))

    # Sort by Y position (top to bottom)
    items.sort(key=lambda x: x[1])

    return items


class DocumentWriter:
    """Write ParsedDocument to Markdown and JSON formats."""

    def __init__(
        self,
        output_dir: Path,
        pdf_name: str,
        header_footer_filter: Optional["HeaderFooterFilter"] = None,
    ):
        """Initialize the document writer.

        Args:
            output_dir: Directory to write output files.
            pdf_name: Base name of the PDF (without extension).
            header_footer_filter: Optional filter for headers/footers.
        """
        self.output_dir = Path(output_dir)
        self.pdf_name = pdf_name
        self.images_dir = self.output_dir / "images"
        self._filter = header_footer_filter

    def write_markdown(self, doc: ParsedDocument, page_height: float = 792.0) -> Path:
        """Write complete Markdown with interleaved content.

        Generates a single Markdown file with all content ordered by page and
        Y position. Tables use to_markdown(), images use get_reference().

        Args:
            doc: ParsedDocument to write.
            page_height: Page height for header/footer filtering.

        Returns:
            Path to the written Markdown file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        lines: List[str] = []

        # Add document title
        lines.append(f"# {doc.title}")
        lines.append("")

        # Process each page
        for page in doc.pages:
            # Add page marker
            lines.append(f"<!-- Page {page.page_number} -->")
            lines.append("")

            # Get interleaved content
            content = interleave_content(
                page.text_blocks,
                page.tables,
                page.images,
            )

            # Collect consecutive text blocks for batch processing
            text_batch: List[TextBlock] = []

            for content_type, _, item in content:
                if content_type == "text":
                    text_batch.append(item)
                else:
                    # Flush text batch before non-text content
                    if text_batch:
                        md_text = blocks_to_markdown(
                            text_batch,
                            filter=self._filter,
                            page_height=page_height,
                            section_markers=doc.section_markers,
                        )
                        if md_text.strip():
                            lines.append(md_text)
                            lines.append("")
                        text_batch = []

                    # Handle non-text content
                    if content_type == "table":
                        table: ExtractedTable = item
                        lines.append(table.to_markdown())
                        lines.append("")
                    elif content_type == "image":
                        image: ExtractedImage = item
                        lines.append(image.get_reference(self.pdf_name))
                        lines.append("")

            # Flush remaining text batch
            if text_batch:
                md_text = blocks_to_markdown(
                    text_batch,
                    filter=self._filter,
                    page_height=page_height,
                    section_markers=doc.section_markers,
                )
                if md_text.strip():
                    lines.append(md_text)
                    lines.append("")

        # Write file
        md_path = self.output_dir / f"{self.pdf_name}.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")

        return md_path

    def write_images(self, doc: ParsedDocument) -> List[str]:
        """Save all images to images_dir.

        Args:
            doc: ParsedDocument containing images.

        Returns:
            List of saved image filenames.
        """
        if not any(page.images for page in doc.pages):
            return []

        self.images_dir.mkdir(parents=True, exist_ok=True)

        filenames: List[str] = []

        for page in doc.pages:
            for image in page.images:
                filename = image.generate_filename(self.pdf_name)
                image_path = self.images_dir / filename
                image_path.write_bytes(image.image_data)
                filenames.append(filename)

        return filenames

    def write_all(self, doc: ParsedDocument, page_height: float = 792.0) -> Tuple[Path, Path, List[str]]:
        """Write Markdown, JSON, and images.

        Args:
            doc: ParsedDocument to write.
            page_height: Page height for header/footer filtering.

        Returns:
            Tuple of (md_path, json_path, image_files).
        """
        from parser.output.json_writer import write_json

        md_path = self.write_markdown(doc, page_height)
        image_files = self.write_images(doc)

        json_path = self.output_dir / f"{self.pdf_name}.json"
        write_json(doc, json_path)

        return md_path, json_path, image_files
