"""Markdown output module for converting TextBlocks to Markdown format.

Converts structured text blocks to Markdown with proper heading levels,
bold/italic formatting, paragraph separation, header/footer filtering,
and section markers for appendices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Set

from parser.extraction.text_extractor import TextBlock
from parser.extraction.section_marker import (
    SectionMarker,
    detect_appendix_sections,
    is_appendix_marker,
)

if TYPE_CHECKING:
    from parser.extraction.header_footer import HeaderFooterFilter


def _escape_markdown(text: str) -> str:
    """Escape special Markdown characters in text.

    Basic escaping to prevent accidental formatting.
    Only escapes characters that could break formatting in body text.

    Args:
        text: Text to escape.

    Returns:
        Escaped text.
    """
    # Escape asterisks and underscores that could trigger formatting
    # Don't escape # at start (headings are handled separately)
    text = text.replace("*", r"\*")
    text = text.replace("_", r"\_")
    return text


def _group_same_line_blocks(blocks: List[TextBlock], y_threshold: float = 5.0) -> List[List[TextBlock]]:
    """Group consecutive blocks that appear on the same line.

    Blocks with similar Y positions (within threshold) are grouped together.

    Args:
        blocks: List of TextBlock objects.
        y_threshold: Maximum Y difference to consider same line (in points).

    Returns:
        List of groups, where each group is a list of TextBlocks on same line.
    """
    if not blocks:
        return []

    groups: List[List[TextBlock]] = []
    current_group: List[TextBlock] = [blocks[0]]
    current_y = blocks[0].bbox[1]  # y0 of first block

    for block in blocks[1:]:
        block_y = block.bbox[1]  # y0

        # Check if this block is on the same line (similar Y position)
        if abs(block_y - current_y) <= y_threshold:
            current_group.append(block)
        else:
            # New line - save current group and start new one
            groups.append(current_group)
            current_group = [block]
            current_y = block_y

    # Don't forget the last group
    if current_group:
        groups.append(current_group)

    return groups


def _format_inline(block: TextBlock, escape: bool = True) -> str:
    """Format a single block as inline Markdown.

    Args:
        block: TextBlock to format.
        escape: Whether to escape special characters.

    Returns:
        Formatted string.
    """
    text = block.text

    if escape:
        text = _escape_markdown(text)

    # Apply bold/italic formatting for body text only
    # (headings don't need inline formatting)
    if block.heading_level == 0:
        if block.is_bold and block.is_italic:
            text = f"***{text}***"
        elif block.is_bold:
            text = f"**{text}**"
        elif block.is_italic:
            text = f"*{text}*"

    return text


def _is_appendix_heading(blocks: List[TextBlock]) -> bool:
    """Check if a group of blocks represents an appendix heading.

    Args:
        blocks: List of TextBlock objects (typically from same line).

    Returns:
        True if this is an appendix heading.
    """
    first_block = blocks[0]
    if first_block.heading_level == 0:
        return False

    # Combine text from all blocks
    combined_text = " ".join(b.text for b in blocks)
    return is_appendix_marker(combined_text, first_block.heading_level)


def blocks_to_markdown(
    blocks: List[TextBlock],
    escape_text: bool = True,
    filter: Optional["HeaderFooterFilter"] = None,
    page_height: float = 792.0,
    section_markers: Optional[List[SectionMarker]] = None,
) -> str:
    """Convert TextBlocks to Markdown string.

    Args:
        blocks: List of TextBlock objects.
        escape_text: Whether to escape special Markdown characters.
        filter: Optional HeaderFooterFilter to skip repetitive content.
        page_height: Page height in points (for filter zone calculation).
        section_markers: Optional list of SectionMarkers for appendix formatting.

    Returns:
        Markdown-formatted string.
    """
    if not blocks:
        return ""

    # Apply filtering if provided
    if filter is not None:
        blocks = [b for b in blocks if not filter.should_skip(b, page_height)]

    if not blocks:
        return ""

    # Build set of appendix heading texts for quick lookup
    appendix_texts: Set[str] = set()
    if section_markers:
        appendix_texts = {m.title for m in section_markers}

    # Group blocks by line
    groups = _group_same_line_blocks(blocks)

    output_lines: List[str] = []
    prev_was_heading = False

    for group in groups:
        # Check if this group contains a heading
        # (heading should be alone on its line typically)
        first_block = group[0]

        if first_block.heading_level > 0:
            # This is a heading
            prefix = "#" * first_block.heading_level
            # Combine all text in the group (in case heading spans multiple spans)
            heading_text = " ".join(b.text for b in group)

            # Check if this is an appendix heading
            is_appendix = _is_appendix_heading(group) or heading_text.strip() in appendix_texts

            if is_appendix:
                # Add horizontal rule before appendix section
                if output_lines:
                    output_lines.append("")
                    output_lines.append("---")
                    output_lines.append("")

                output_lines.append(f"{prefix} {heading_text}")
                output_lines.append("*[Appendix section]*")
            else:
                output_lines.append(f"{prefix} {heading_text}")

            prev_was_heading = True
        else:
            # This is body text
            # Combine spans on same line
            line_parts = []
            for block in group:
                formatted = _format_inline(block, escape=escape_text)
                line_parts.append(formatted)

            line_text = "".join(line_parts)

            # Add blank line before paragraph if previous was heading
            # (Markdown needs blank line after heading for proper rendering)
            if prev_was_heading and output_lines:
                output_lines.append("")

            output_lines.append(line_text)
            prev_was_heading = False

    # Join with newlines, adding blank lines between paragraphs
    # First pass: join lines
    result_lines: List[str] = []
    for i, line in enumerate(output_lines):
        result_lines.append(line)

        # Add blank line after non-empty body lines (paragraph separation)
        # but not after headings (handled above) or empty lines
        if line and not line.startswith("#") and not line.startswith("*[") and not line.startswith("---") and i < len(output_lines) - 1:
            next_line = output_lines[i + 1]
            # If next line is also body text (not heading, not empty), add separator
            if next_line and not next_line.startswith("#") and not next_line.startswith("---"):
                # Check if there's already a blank line
                if result_lines and result_lines[-1] != "":
                    pass  # Don't add double blanks

    return "\n".join(result_lines)


def page_to_markdown(blocks: List[TextBlock], page_number: int) -> str:
    """Convert a page's TextBlocks to Markdown with page marker.

    Args:
        blocks: List of TextBlock objects from a single page.
        page_number: Page number (1-indexed).

    Returns:
        Markdown string with page comment header.
    """
    content = blocks_to_markdown(blocks)

    # Add page marker comment
    return f"<!-- Page {page_number} -->\n\n{content}"


class MarkdownWriter:
    """Writer for multi-page Markdown documents with filtering and section detection.

    Collects pages and produces a combined Markdown document.
    Supports header/footer filtering and appendix section detection.
    """

    def __init__(self, filter: Optional["HeaderFooterFilter"] = None):
        """Initialize the writer.

        Args:
            filter: Optional HeaderFooterFilter to skip repetitive content.
        """
        self._pages: List[tuple[int, str]] = []
        self._filter = filter
        self._section_markers: List[SectionMarker] = []

    def add_page(
        self,
        blocks: List[TextBlock],
        page_number: int,
        page_height: float = 792.0,
    ) -> None:
        """Add a page to the document.

        Args:
            blocks: TextBlock objects from the page.
            page_number: Page number (1-indexed).
            page_height: Page height in points (for filter zone calculation).
        """
        # Detect section markers in this page's blocks
        markers = detect_appendix_sections(blocks)
        self._section_markers.extend(markers)

        # Convert to markdown with filtering
        content = blocks_to_markdown(
            blocks,
            filter=self._filter,
            page_height=page_height,
            section_markers=markers,
        )
        self._pages.append((page_number, content))

    def get_markdown(self) -> str:
        """Get the complete Markdown document.

        Returns:
            Combined Markdown string with page markers.
        """
        if not self._pages:
            return ""

        parts: List[str] = []
        for page_num, content in self._pages:
            parts.append(f"<!-- Page {page_num} -->")
            parts.append("")
            parts.append(content)
            parts.append("")  # Blank line between pages

        return "\n".join(parts).rstrip() + "\n"

    def get_section_markers(self) -> List[SectionMarker]:
        """Get all section markers detected across pages.

        Returns:
            List of SectionMarker objects.
        """
        return self._section_markers.copy()

    @property
    def page_count(self) -> int:
        """Get the number of pages added."""
        return len(self._pages)

    @property
    def filter(self) -> Optional["HeaderFooterFilter"]:
        """Get the filter used by this writer."""
        return self._filter
