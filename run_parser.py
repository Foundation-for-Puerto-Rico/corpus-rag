#!/usr/bin/env python3
"""Main script to process WCRP PDF documents.

Processes all PDFs in a directory, extracting text, tables, and images,
then generates Markdown and JSON output files.

Usage:
    python run_parser.py "planes pdf" output
    python run_parser.py "planes pdf" output --single "4ta_Seccion_Levittown.pdf"
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from parser.detection import PageType, detect_page_type
from parser.ocr import ocr_page
from parser.extraction.text_extractor import TextExtractor, TextBlock
from parser.extraction.table_extractor import extract_tables_from_page
from parser.extraction.image_extractor import extract_images_from_page
from parser.extraction.section_marker import detect_all_sections
from parser.extraction.header_footer import build_filter
from parser.output.document import ParsedDocument, ParsedPage, DocumentWriter


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    verbose: bool = True,
) -> tuple[Path, Path, int]:
    """Process a single PDF and generate outputs.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output files.
        verbose: Print progress messages.

    Returns:
        Tuple of (markdown_path, json_path, page_count).
    """
    pdf_name = pdf_path.stem
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_name}")
        print(f"{'='*60}")

    start_time = time.time()

    # Open document
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if verbose:
        print(f"  Pages: {total_pages}")

    # Build extractors and filters
    text_extractor = TextExtractor(doc)
    header_footer_filter = build_filter(doc)

    # Track statistics
    scanned_pages = 0
    native_pages = 0

    # Process pages
    pages: List[ParsedPage] = []
    all_text_blocks: List[TextBlock] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1

        if verbose and page_num % 10 == 0:
            print(f"  Page {page_num}/{total_pages}...")

        # Detect page type
        page_type = detect_page_type(page)

        # Extract text based on page type
        if page_type == PageType.SCANNED:
            # Use OCR for scanned pages
            ocr_result = ocr_page(page, language="spa", page_number=page_num)
            # For scanned pages, we can't get structured text blocks easily
            # Create a single text block from OCR text
            if ocr_result.text.strip():
                text_blocks = [TextBlock(
                    text=ocr_result.text,
                    heading_level=0,
                    is_bold=False,
                    is_italic=False,
                    font_size=12.0,
                    bbox=(72, 72, 540, 720),  # Approximate full page
                    page_number=page_num,
                )]
            else:
                text_blocks = []
            scanned_pages += 1
        elif page_type == PageType.EMPTY:
            text_blocks = []
            native_pages += 1
        else:
            # Native or mixed - extract structured text
            text_blocks = text_extractor.extract_page(page, page_num)
            native_pages += 1

        all_text_blocks.extend(text_blocks)

        # Extract tables
        tables = extract_tables_from_page(page, page_num)

        # Extract images
        images = extract_images_from_page(doc, page, page_num)

        # Create parsed page
        parsed_page = ParsedPage(
            page_number=page_num,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
        )
        pages.append(parsed_page)

        # Memory cleanup
        gc.collect()

    # Detect section markers from all text blocks
    section_markers = detect_all_sections(all_text_blocks)

    # Determine document title (first h1 heading or filename)
    title = pdf_name
    for block in all_text_blocks:
        if block.heading_level == 1:
            title = block.text.strip()
            break

    # Build ParsedDocument
    duration = time.time() - start_time
    parsed_doc = ParsedDocument(
        source_pdf=pdf_path.name,
        title=title,
        total_pages=total_pages,
        pages=pages,
        section_markers=section_markers,
        scanned_pages=scanned_pages,
        native_pages=native_pages,
        processing_time_seconds=duration,
    )

    # Close document
    doc.close()

    # Write outputs
    writer = DocumentWriter(
        output_dir,
        pdf_name,
        header_footer_filter=header_footer_filter,
    )

    # Get page height for header/footer filtering (assume letter size)
    page_height = 792.0

    md_path, json_path, image_files = writer.write_all(parsed_doc, page_height)

    if verbose:
        print(f"  Scanned pages: {scanned_pages}")
        print(f"  Native pages: {native_pages}")
        print(f"  Tables found: {sum(len(p.tables) for p in pages)}")
        print(f"  Images found: {sum(len(p.images) for p in pages)}")
        print(f"  Sections: {len(section_markers)}")
        print(f"  Time: {duration:.1f}s")
        print(f"  Output: {md_path.name}, {json_path.name}")
        if image_files:
            print(f"  Images saved: {len(image_files)}")

    return md_path, json_path, total_pages


def process_directory(
    input_dir: Path,
    output_dir: Path,
    single_file: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Process all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory for output files.
        single_file: If provided, only process this file.
        verbose: Print progress messages.

    Returns:
        Dictionary with processing statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find PDFs
    if single_file:
        pdf_files = [input_dir / single_file]
        if not pdf_files[0].exists():
            print(f"Error: File not found: {pdf_files[0]}")
            sys.exit(1)
    else:
        pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        sys.exit(1)

    if verbose:
        print(f"\nFound {len(pdf_files)} PDF(s) to process")
        print(f"Output directory: {output_dir}")

    # Process each PDF
    start_time = time.time()
    results = {
        "total_pdfs": len(pdf_files),
        "completed": 0,
        "failed": 0,
        "total_pages": 0,
        "failures": {},
    }

    for i, pdf_path in enumerate(pdf_files, 1):
        if verbose:
            print(f"\n[{i}/{len(pdf_files)}] ", end="")

        try:
            _, _, page_count = process_single_pdf(pdf_path, output_dir, verbose)
            results["completed"] += 1
            results["total_pages"] += page_count
        except Exception as e:
            results["failed"] += 1
            results["failures"][str(pdf_path)] = str(e)
            if verbose:
                print(f"  ERROR: {e}")

    # Final summary
    duration = time.time() - start_time
    results["duration_seconds"] = duration

    if verbose:
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  PDFs processed: {results['completed']}/{results['total_pdfs']}")
        print(f"  Total pages: {results['total_pages']}")
        print(f"  Total time: {duration:.1f}s")
        if results["failed"] > 0:
            print(f"  Failed: {results['failed']}")
            for path, error in results["failures"].items():
                print(f"    - {Path(path).name}: {error}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process WCRP PDF documents and generate Markdown/JSON output."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "output_dir",
        help="Directory for output files",
    )
    parser.add_argument(
        "--single",
        "-s",
        dest="single_file",
        help="Process only this file (filename within input_dir)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        sys.exit(1)

    process_directory(
        input_dir,
        output_dir,
        single_file=args.single_file,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
