"""Table extraction module using PyMuPDF's find_tables() API.

Extracts tables from PDF pages with structure metadata including
bounding boxes, row/column counts, and header names.
"""

from __future__ import annotations

from typing import Generator, List, Optional

import fitz  # PyMuPDF

from parser.models.table import ExtractedTable

# Constants for borderless table validation
MIN_BORDERLESS_ROWS = 2
MIN_BORDERLESS_COLS = 2
MIN_WORDS_VERTICAL = 3
MIN_WORDS_HORIZONTAL = 3


def fill_merged_cells(
    data: List[List[Optional[str]]],
    horizontal: bool = True,
    vertical: bool = True,
) -> List[List[Optional[str]]]:
    """Fill None values representing merged cells using forward-fill.

    WARNING: Heuristic-based. Forward-fill may not be correct for all
    merged cell patterns. Tables with high merge percentage (>30%)
    should be flagged for manual review.

    Args:
        data: Table data with None values from PyMuPDF.
        horizontal: Forward-fill across columns (left to right).
        vertical: Forward-fill down rows (top to bottom).

    Returns:
        Table data with None values filled where possible.
    """
    if not data:
        return data

    result = [list(row) for row in data]  # Deep copy

    # Forward fill across columns (horizontal merge)
    if horizontal:
        for row in result:
            for i in range(1, len(row)):
                if row[i] is None and row[i - 1] is not None:
                    row[i] = row[i - 1]

    # Forward fill down rows (vertical merge)
    if vertical:
        for col in range(len(result[0]) if result else 0):
            for row_idx in range(1, len(result)):
                if result[row_idx][col] is None and result[row_idx - 1][col] is not None:
                    result[row_idx][col] = result[row_idx - 1][col]

    return result


def _is_valid_borderless_table(data: List[List[Optional[str]]]) -> bool:
    """Validate that detected borderless table is actually tabular.

    Prevents false positives from text strategy detecting normal paragraphs.

    Args:
        data: Raw extracted table data.

    Returns:
        True if likely a real table, False if probably false positive.
    """
    if not data:
        return False

    # Must have at least 2x2
    if len(data) < MIN_BORDERLESS_ROWS:
        return False
    if not data[0] or len(data[0]) < MIN_BORDERLESS_COLS:
        return False

    # Check for excessive None values (>80% = probably not a table)
    total_cells = sum(len(row) for row in data)
    non_empty = sum(1 for row in data for cell in row if cell and cell.strip())
    if non_empty / total_cells < 0.2:
        return False

    # Check for consistent column count (tables have uniform structure)
    col_counts = [len(row) for row in data]
    if max(col_counts) - min(col_counts) > 2:
        return False

    return True


def extract_tables_from_page(
    page: fitz.Page,
    page_number: int,
    try_borderless: bool = True,
) -> List[ExtractedTable]:
    """Extract all tables from a page with optional borderless fallback.

    Uses PyMuPDF's find_tables() with default strategy for bordered tables.
    If no bordered tables found and try_borderless=True, falls back to
    text strategy for borderless tables with validation.

    Args:
        page: PyMuPDF Page object.
        page_number: 1-indexed page number.
        try_borderless: If True and no bordered tables found, try text strategy.

    Returns:
        List of ExtractedTable objects found on the page.
    """
    tables: List[ExtractedTable] = []

    # First pass: default strategy (bordered tables)
    finder = page.find_tables()

    for i, tab in enumerate(finder.tables):
        # Extract raw data
        data = tab.extract()

        # Get header names from table.header if available
        # tab.header.names may be None or a list
        header_names_raw = tab.header.names if tab.header and tab.header.names else None

        # If header.names is None or empty, use first row as headers
        if header_names_raw:
            header_names = [str(h) if h is not None else "" for h in header_names_raw]
        elif data:
            header_names = [str(cell) if cell is not None else "" for cell in data[0]]
        else:
            header_names = []

        tables.append(
            ExtractedTable(
                page_number=page_number,
                table_index=i,
                bbox=tuple(tab.bbox),
                rows=data,
                row_count=tab.row_count,
                col_count=tab.col_count,
                header_names=header_names,
                is_borderless=False,
            )
        )

    # Second pass: text strategy for borderless tables (only if no bordered found)
    if not tables and try_borderless:
        finder = page.find_tables(
            strategy="text",
            min_words_vertical=MIN_WORDS_VERTICAL,
            min_words_horizontal=MIN_WORDS_HORIZONTAL,
        )

        for i, tab in enumerate(finder.tables):
            data = tab.extract()

            # Validate to avoid false positives
            if not _is_valid_borderless_table(data):
                continue

            header_names_raw = tab.header.names if tab.header and tab.header.names else None

            if header_names_raw:
                header_names = [str(h) if h is not None else "" for h in header_names_raw]
            elif data:
                header_names = [str(cell) if cell is not None else "" for cell in data[0]]
            else:
                header_names = []

            tables.append(
                ExtractedTable(
                    page_number=page_number,
                    table_index=i,
                    bbox=tuple(tab.bbox),
                    rows=data,
                    row_count=tab.row_count,
                    col_count=tab.col_count,
                    header_names=header_names,
                    is_borderless=True,
                )
            )

    return tables


class TableExtractor:
    """Extract tables from multi-page PDF documents.

    Similar to TextExtractor pattern for consistency.
    """

    def __init__(self, doc: fitz.Document, try_borderless: bool = True):
        """Initialize extractor with a PDF document.

        Args:
            doc: PyMuPDF Document object.
            try_borderless: If True, fall back to text strategy when no bordered
                tables found (default True).
        """
        self.doc = doc
        self.try_borderless = try_borderless

    def extract_page(self, page_number: int) -> List[ExtractedTable]:
        """Extract tables from a single page (1-indexed).

        Args:
            page_number: Page number (1-indexed).

        Returns:
            List of ExtractedTable objects from the page.
        """
        page = self.doc[page_number - 1]
        return extract_tables_from_page(page, page_number, self.try_borderless)

    def extract_all(self) -> Generator[ExtractedTable, None, None]:
        """Extract tables from all pages as generator (memory efficient).

        Yields:
            ExtractedTable objects from each page in document order.
        """
        for page_num in range(len(self.doc)):
            tables = extract_tables_from_page(
                self.doc[page_num], page_num + 1, self.try_borderless
            )
            yield from tables

    def get_table_count(self) -> int:
        """Get total table count across document.

        Returns:
            Total number of tables in the document.
        """
        return sum(1 for _ in self.extract_all())
