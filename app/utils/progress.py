"""
Progress tracking utilities for long-running tasks.
"""

import json
import os
from pathlib import Path
from .logging import setup_logging

logger = setup_logging()

# Ensure progress directory exists
APP_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROGRESS_DIR = APP_DIR / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)


def save_progress(job_id, total_or_data, processed=None, success=True, error=None):
    """
    Saves job progress to a file.
    
    Args:
        job_id (str): Unique job identifier
        total_or_data: Either the total number of items to process, or a dictionary with progress data
        processed (int, optional): Number of items processed so far (only used if total_or_data is numeric)
        success (bool, optional): Whether the job is successful so far
        error (str, optional): Error message if any
    
    Returns:
        dict: Progress data
    """
    progress_file = PROGRESS_DIR / f"{job_id}_progress.json"
    
    # Check if using dictionary format or numeric format
    if isinstance(total_or_data, dict):
        # Dictionary format
        custom_data = total_or_data
        # Extract standard fields if present
        total = custom_data.get("total", 100)
        processed = custom_data.get("processed", 0)
        # Use progress percentage from custom data if available
        if "progress" in custom_data:
            percentage = float(custom_data["progress"])
        else:
            percentage = (processed / total) * 100 if total > 0 else 0
        # Completion status from custom data
        completion_status = custom_data.get("completed", processed >= total)
        # Error information
        error_msg = custom_data.get("error", error)
        success_status = custom_data.get("success", success)
    else:
        # Numeric format (original)
        total = total_or_data
        # Calculate percentage with some minimum threshold to show movement
        if processed > 0 and processed < total:
            # Ensure we show at least 1% progress and never more than 99% until complete
            percentage = max(1, min(99, round((processed / total) * 100 if total > 0 else 0, 1)))
        else:
            percentage = 100 if processed >= total else 0
        # Standard completion status
        completion_status = processed >= total
        error_msg = error
        success_status = success
        custom_data = {}
    
    # Update existing file if it exists to maintain history
    existing_data = {}
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Only increase percentage, never decrease (prevents jumps backward)
                if 'percentage' in existing_data and existing_data['percentage'] > percentage and not completion_status:
                    percentage = existing_data['percentage']
        except Exception:
            pass
    
    # Start with basic progress data
    progress_data = {
        "job_id": job_id,
        "total": total,
        "processed": processed,
        "percentage": percentage,
        "success": success_status,
        "error": error_msg,
        "completed": completion_status,
        "timestamp": os.path.getmtime(progress_file) if progress_file.exists() else None
    }
    
    # Add any custom fields from dictionary data
    if isinstance(total_or_data, dict):
        for key, value in total_or_data.items():
            if key not in ["job_id"]:  # Don't override job_id
                progress_data[key] = value
    
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
        # Update timestamp after writing
        progress_data["timestamp"] = os.path.getmtime(progress_file)
    except Exception as e:
        logger.error(f"Error saving progress: {e}")
    
    return progress_data


def get_progress(job_id):
    """
    Gets job progress information.
    
    Args:
        job_id (str): Unique job identifier
    
    Returns:
        dict: Progress data or None if not found
    """
    progress_file = PROGRESS_DIR / f"{job_id}_progress.json"
    
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading progress: {e}")
        return None