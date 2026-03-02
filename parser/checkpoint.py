"""Checkpoint and resume functionality for batch PDF processing.

Tracks processing progress to enable resuming interrupted batch operations.
Uses JSON format for human-readable checkpoint files.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class Checkpoint:
    """Represents the state of a batch processing operation.

    Attributes:
        started_at: ISO timestamp when batch started
        updated_at: ISO timestamp of last update
        total_pdfs: Total number of PDFs in the batch
        completed_pdfs: List of paths to fully processed PDFs
        current_pdf: Path to PDF currently being processed (None if between PDFs)
        current_page: Last completed page number in current PDF (0 if not started)
        failed_pdfs: Dict mapping failed PDF paths to error messages
    """

    started_at: str
    updated_at: str
    total_pdfs: int
    completed_pdfs: List[str] = field(default_factory=list)
    current_pdf: Optional[str] = None
    current_page: int = 0
    failed_pdfs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Checkpoint:
        """Create Checkpoint from dictionary."""
        return cls(
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            total_pdfs=data["total_pdfs"],
            completed_pdfs=data.get("completed_pdfs", []),
            current_pdf=data.get("current_pdf"),
            current_page=data.get("current_page", 0),
            failed_pdfs=data.get("failed_pdfs", {}),
        )


class CheckpointManager:
    """Manages checkpoint persistence and resume logic.

    Provides atomic file writes to prevent corruption from interrupted saves.
    Saves checkpoint after every page for fine-grained resume capability.
    """

    def __init__(self, checkpoint_path: Union[str, Path]):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path where checkpoint file will be stored
        """
        self.path = Path(checkpoint_path)
        self.checkpoint: Optional[Checkpoint] = None
        self._all_pdfs: List[str] = []

    def load(self) -> Optional[Checkpoint]:
        """Load checkpoint from disk if it exists.

        Returns:
            Loaded Checkpoint or None if file doesn't exist

        Raises:
            json.JSONDecodeError: If checkpoint file is corrupted
        """
        if not self.path.exists():
            return None

        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.checkpoint = Checkpoint.from_dict(data)
        self._all_pdfs = data.get("_all_pdfs", [])
        return self.checkpoint

    def save(self) -> None:
        """Persist checkpoint to disk atomically.

        Uses write-to-temp-then-rename pattern to prevent corruption
        from interrupted writes (atomic on POSIX systems).
        """
        if self.checkpoint is None:
            return

        # Update timestamp
        self.checkpoint.updated_at = datetime.utcnow().isoformat() + "Z"

        # Prepare data
        data = self.checkpoint.to_dict()
        data["_all_pdfs"] = self._all_pdfs

        # Atomic write: write to temp file, then rename
        # Create temp file in same directory to ensure same filesystem
        parent_dir = self.path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="checkpoint_",
            dir=str(parent_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename (on POSIX)
            os.replace(temp_path, self.path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def start_batch(self, pdf_paths: List[str]) -> Checkpoint:
        """Initialize a new checkpoint for a batch operation.

        Args:
            pdf_paths: List of PDF paths to process

        Returns:
            Newly created Checkpoint
        """
        now = datetime.utcnow().isoformat() + "Z"
        self.checkpoint = Checkpoint(
            started_at=now,
            updated_at=now,
            total_pdfs=len(pdf_paths),
            completed_pdfs=[],
            current_pdf=None,
            current_page=0,
            failed_pdfs={},
        )
        self._all_pdfs = list(pdf_paths)
        return self.checkpoint

    def start_pdf(self, pdf_path: str) -> None:
        """Mark a PDF as currently being processed.

        Args:
            pdf_path: Path to the PDF being started
        """
        if self.checkpoint is None:
            raise RuntimeError("No checkpoint active. Call start_batch first.")

        self.checkpoint.current_pdf = pdf_path
        self.checkpoint.current_page = 0

    def complete_page(self, page_num: int) -> None:
        """Update progress within current PDF.

        Args:
            page_num: The page number that was just completed (1-indexed)
        """
        if self.checkpoint is None:
            raise RuntimeError("No checkpoint active. Call start_batch first.")

        self.checkpoint.current_page = page_num

    def complete_pdf(self) -> None:
        """Mark the current PDF as fully completed."""
        if self.checkpoint is None:
            raise RuntimeError("No checkpoint active. Call start_batch first.")

        if self.checkpoint.current_pdf is not None:
            if self.checkpoint.current_pdf not in self.checkpoint.completed_pdfs:
                self.checkpoint.completed_pdfs.append(self.checkpoint.current_pdf)
            self.checkpoint.current_pdf = None
            self.checkpoint.current_page = 0

    def fail_pdf(self, error: str) -> None:
        """Record a PDF failure and continue processing.

        Args:
            error: Error message describing the failure
        """
        if self.checkpoint is None:
            raise RuntimeError("No checkpoint active. Call start_batch first.")

        if self.checkpoint.current_pdf is not None:
            self.checkpoint.failed_pdfs[self.checkpoint.current_pdf] = error
            self.checkpoint.current_pdf = None
            self.checkpoint.current_page = 0

    def get_pending_pdfs(self) -> List[str]:
        """Get list of PDFs not yet completed.

        Returns:
            List of PDF paths that have not been marked as completed
        """
        if self.checkpoint is None:
            return self._all_pdfs.copy()

        completed = set(self.checkpoint.completed_pdfs)
        failed = set(self.checkpoint.failed_pdfs.keys())
        finished = completed | failed

        return [p for p in self._all_pdfs if p not in finished]

    def get_resume_page(self, pdf_path: str) -> int:
        """Get the page number to resume from for a given PDF.

        Args:
            pdf_path: Path to the PDF

        Returns:
            Page number to resume from (0 if not started, current_page + 1 if mid-PDF)
        """
        if self.checkpoint is None:
            return 0

        # If this is the current PDF being processed, resume from next page
        if self.checkpoint.current_pdf == pdf_path:
            return self.checkpoint.current_page + 1

        # If completed, return -1 to indicate skip
        if pdf_path in self.checkpoint.completed_pdfs:
            return -1

        # If failed, return -1 to indicate skip
        if pdf_path in self.checkpoint.failed_pdfs:
            return -1

        # Not started yet
        return 0

    def is_complete(self) -> bool:
        """Check if all PDFs have been processed (completed or failed).

        Returns:
            True if no pending PDFs remain
        """
        return len(self.get_pending_pdfs()) == 0

    def get_stats(self) -> dict:
        """Get summary statistics for the batch.

        Returns:
            Dict with completed, failed, pending, and total counts
        """
        if self.checkpoint is None:
            return {
                "total": 0,
                "completed": 0,
                "failed": 0,
                "pending": 0,
            }

        pending = len(self.get_pending_pdfs())
        return {
            "total": self.checkpoint.total_pdfs,
            "completed": len(self.checkpoint.completed_pdfs),
            "failed": len(self.checkpoint.failed_pdfs),
            "pending": pending,
        }
