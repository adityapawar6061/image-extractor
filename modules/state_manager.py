"""State manager for persisting processing state to disk for resume capability."""

import json
import os
import time
from pathlib import Path
from typing import Any

STATE_FILE = "processing_state.json"
RESULTS_DIR = Path("data/results")
FAILED_DIR = Path("data/failed")


def get_state_path() -> Path:
    """Return the path to the processing state file."""
    return Path(STATE_FILE)


def load_state() -> dict[str, Any]:
    """Load the processing state from disk."""
    path = get_state_path()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "images": {},
        "total_processed": 0,
        "total_failed": 0,
        "total_rows": 0,
        "started_at": None,
        "last_updated": None,
        "columns": [],
        "custom_instructions": "",
    }


def save_state(state: dict[str, Any]) -> None:
    """Save the processing state to disk."""
    state["last_updated"] = time.time()
    path = get_state_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def init_image_state(filename: str) -> dict[str, Any]:
    """Create initial state entry for a single image."""
    return {
        "filename": filename,
        "status": "pending",  # pending | processing | completed | failed | needs_review
        "rows_extracted": 0,
        "result": [],
        "error": None,
        "retry_count": 0,
        "timestamp": None,
        "needs_review": False,
        "review_reasons": [],
    }


def update_image_state(
    state: dict[str, Any],
    filename: str,
    *,
    status: str | None = None,
    rows_extracted: int | None = None,
    result: list[dict] | None = None,
    error: str | None = None,
    retry_count: int | None = None,
    needs_review: bool | None = None,
    review_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Update the state entry for a single image."""
    if filename not in state["images"]:
        state["images"][filename] = init_image_state(filename)

    img = state["images"][filename]
    if status is not None:
        img["status"] = status
    if rows_extracted is not None:
        img["rows_extracted"] = rows_extracted
    if result is not None:
        img["result"] = result
    if error is not None:
        img["error"] = error
    if retry_count is not None:
        img["retry_count"] = retry_count
    if needs_review is not None:
        img["needs_review"] = needs_review
    if review_reasons is not None:
        img["review_reasons"] = review_reasons

    img["timestamp"] = time.time()

    # Update aggregate counters
    state["total_processed"] = sum(
        1 for v in state["images"].values() if v["status"] == "completed"
    )
    state["total_failed"] = sum(
        1 for v in state["images"].values() if v["status"] == "failed"
    )
    state["total_rows"] = sum(
        v["rows_extracted"] for v in state["images"].values()
    )

    return state


def get_completed_filenames(state: dict[str, Any]) -> set[str]:
    """Return set of filenames that have been successfully processed."""
    return {
        name
        for name, info in state["images"].items()
        if info["status"] == "completed"
    }


def get_failed_filenames(state: dict[str, Any]) -> set[str]:
    """Return set of filenames that failed processing."""
    return {
        name
        for name, info in state["images"].items()
        if info["status"] == "failed"
    }


def get_all_results(state: dict[str, Any]) -> list[dict]:
    """Gather all extracted rows from all completed images."""
    all_rows = []
    for name, info in state["images"].items():
        if info["status"] in ("completed", "needs_review") and info.get("result"):
            for row in info["result"]:
                row["source_image"] = name
                all_rows.append(row)
    return all_rows


def get_needs_review_results(state: dict[str, Any]) -> list[dict]:
    """Return results that need human review."""
    all_rows = []
    for name, info in state["images"].items():
        if info.get("needs_review") and info.get("result"):
            for row in info["result"]:
                row["source_image"] = name
                all_rows.append(row)
    return all_rows


def get_failed_images(state: dict[str, Any]) -> list[dict]:
    """Return list of failed image records."""
    results = []
    for name, info in state["images"].items():
        if info["status"] == "failed":
            results.append(
                {
                    "filename": name,
                    "error": info.get("error", "Unknown"),
                    "retry_count": info.get("retry_count", 0),
                    "timestamp": info.get("timestamp"),
                }
            )
    return results


def clear_state() -> None:
    """Delete the state file."""
    path = get_state_path()
    if path.exists():
        path.unlink()


def reset_for_new_job(state: dict[str, Any]) -> dict[str, Any]:
    """Reset state for a new job, clearing previous image data."""
    state["images"] = {}
    state["total_processed"] = 0
    state["total_failed"] = 0
    state["total_rows"] = 0
    state["started_at"] = time.time()
    state["last_updated"] = None
    return state


def save_result_json(filename: str, rows: list[dict]) -> None:
    """Save extracted rows to a per-image JSON file in data/results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{Path(filename).stem}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def save_failed_log(filename: str, error: str, retry_count: int) -> None:
    """Append a failure record to the failed-images log."""
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    log_file = FAILED_DIR / "failed_images.json"
    entries = []
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    entries.append(
        {
            "filename": filename,
            "error_message": error,
            "retry_count": retry_count,
            "timestamp": time.time(),
        }
    )
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
