"""Batch processing for multiple PDF documents.

Processes multiple PDFs with checkpoint/resume capability.
Designed for handling large batches (33+ PDFs) without manual intervention.
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

from parser.checkpoint import CheckpointManager
from parser.processor import PDFProcessor, ProcessedPage


@dataclass
class BatchResult:
    """Summary of a batch processing operation.

    Attributes:
        total_pdfs: Total number of PDFs in the batch
        completed_pdfs: Number of PDFs successfully processed
        failed_pdfs: Number of PDFs that failed
        total_pages: Total pages processed across all PDFs
        scanned_pages: Pages that required OCR
        native_pages: Pages with native text extraction
        duration_seconds: Total processing time in seconds
        failures: Dict mapping failed PDF paths to error messages
    """

    total_pdfs: int
    completed_pdfs: int
    failed_pdfs: int
    total_pages: int
    scanned_pages: int
    native_pages: int
    duration_seconds: float
    failures: Dict[str, str] = field(default_factory=dict)


class BatchProcessor:
    """Processes multiple PDFs with checkpoint/resume capability.

    Features:
    - Automatic checkpoint after every page (fine-grained resume)
    - Failure handling: one PDF failure doesn't stop the batch
    - Generator-based streaming for memory efficiency
    - Progress callbacks for monitoring

    Example:
        processor = BatchProcessor()
        for pdf_path, page in processor.process_directory("./pdfs"):
            # Handle each page as it's processed
            print(f"{pdf_path} page {page.page_number}: {page.page_type}")
    """

    def __init__(
        self,
        checkpoint_path: str = ".corpus_checkpoint.json",
        language: str = "spa",
        on_pdf_complete: Optional[Callable[[str, int], None]] = None,
        on_page_complete: Optional[Callable[[str, int, ProcessedPage], None]] = None,
    ):
        """Initialize the batch processor.

        Args:
            checkpoint_path: Path to checkpoint file (default: .corpus_checkpoint.json)
            language: Tesseract language code for OCR (default: 'spa' for Spanish)
            on_pdf_complete: Callback(pdf_path, page_count) called when a PDF finishes
            on_page_complete: Callback(pdf_path, page_num, page) called for each page
        """
        self.checkpoint_mgr = CheckpointManager(checkpoint_path)
        self.processor = PDFProcessor(language=language)
        self.on_pdf_complete = on_pdf_complete
        self.on_page_complete = on_page_complete

    def process_batch(
        self,
        pdf_paths: List[str],
        resume: bool = True,
    ) -> Generator[Tuple[str, ProcessedPage], None, BatchResult]:
        """Process multiple PDFs, yielding pages as they're processed.

        Yields (pdf_path, ProcessedPage) tuples for each page.
        Returns BatchResult with statistics when complete.

        Args:
            pdf_paths: List of PDF file paths to process
            resume: If True, resume from checkpoint if one exists (default: True)

        Yields:
            Tuple of (pdf_path, ProcessedPage) for each processed page

        Returns:
            BatchResult with processing statistics (via generator return)
        """
        start_time = time.time()

        # Statistics tracking
        total_pages = 0
        scanned_pages = 0
        native_pages = 0
        pdf_page_counts: Dict[str, int] = {}

        # Load or create checkpoint
        if resume and self.checkpoint_mgr.path.exists():
            self.checkpoint_mgr.load()
            # Keep existing state but update all_pdfs to include new paths
            self.checkpoint_mgr._all_pdfs = list(pdf_paths)
        else:
            self.checkpoint_mgr.start_batch(pdf_paths)

        # Get PDFs that still need processing
        pending_pdfs = self.checkpoint_mgr.get_pending_pdfs()

        for pdf_path in pending_pdfs:
            try:
                # Check if file exists
                if not Path(pdf_path).exists():
                    self.checkpoint_mgr.start_pdf(pdf_path)
                    self.checkpoint_mgr.fail_pdf(f"File not found: {pdf_path}")
                    self.checkpoint_mgr.save()
                    continue

                # Get resume point BEFORE calling start_pdf
                # (start_pdf resets current_page to 0)
                resume_page = self.checkpoint_mgr.get_resume_page(pdf_path)

                # Start processing this PDF
                self.checkpoint_mgr.start_pdf(pdf_path)

                # Track pages for this PDF
                pdf_page_count = 0

                # Process pages
                for processed_page in self.processor.process_pdf(pdf_path):
                    # Skip pages that were already processed (resume_page is 1-indexed next page)
                    if processed_page.page_number < resume_page:
                        continue

                    # Update statistics
                    total_pages += 1
                    pdf_page_count += 1

                    if processed_page.ocr_applied:
                        scanned_pages += 1
                    else:
                        native_pages += 1

                    # Update checkpoint after each page
                    self.checkpoint_mgr.complete_page(processed_page.page_number)
                    self.checkpoint_mgr.save()

                    # Invoke page callback if provided
                    if self.on_page_complete:
                        self.on_page_complete(
                            pdf_path,
                            processed_page.page_number,
                            processed_page,
                        )

                    # Yield the page
                    yield (pdf_path, processed_page)

                # PDF completed successfully
                self.checkpoint_mgr.complete_pdf()
                self.checkpoint_mgr.save()

                pdf_page_counts[pdf_path] = pdf_page_count

                # Invoke PDF completion callback if provided
                if self.on_pdf_complete:
                    self.on_pdf_complete(pdf_path, pdf_page_count)

            except Exception as e:
                # Record failure and continue to next PDF
                error_msg = str(e)
                self.checkpoint_mgr.fail_pdf(error_msg)
                self.checkpoint_mgr.save()

        # Calculate duration
        duration = time.time() - start_time

        # Build final result
        stats = self.checkpoint_mgr.get_stats()

        return BatchResult(
            total_pdfs=stats["total"],
            completed_pdfs=stats["completed"],
            failed_pdfs=stats["failed"],
            total_pages=total_pages,
            scanned_pages=scanned_pages,
            native_pages=native_pages,
            duration_seconds=duration,
            failures=dict(self.checkpoint_mgr.checkpoint.failed_pdfs),
        )

    def process_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        resume: bool = True,
    ) -> Generator[Tuple[str, ProcessedPage], None, BatchResult]:
        """Process all PDFs matching pattern in a directory.

        Args:
            directory: Directory to search for PDFs
            pattern: Glob pattern for PDF files (default: "*.pdf")
            resume: If True, resume from checkpoint if one exists (default: True)

        Yields:
            Tuple of (pdf_path, ProcessedPage) for each processed page

        Returns:
            BatchResult with processing statistics (via generator return)
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all PDFs matching pattern
        search_pattern = str(directory / pattern)
        pdf_paths = sorted(glob.glob(search_pattern))

        if not pdf_paths:
            # Return empty result if no PDFs found
            return BatchResult(
                total_pdfs=0,
                completed_pdfs=0,
                failed_pdfs=0,
                total_pages=0,
                scanned_pages=0,
                native_pages=0,
                duration_seconds=0.0,
                failures={},
            )

        # Delegate to process_batch
        return (yield from self.process_batch(pdf_paths, resume=resume))

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint file to start fresh.

        Use this to reset progress and start processing from the beginning.
        """
        if self.checkpoint_mgr.path.exists():
            self.checkpoint_mgr.path.unlink()
