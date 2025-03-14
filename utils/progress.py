"""
Progress tracking utilities for long-running tasks.
"""

import json
import os
from pathlib import Path
from .logging import setup_logging

logger = setup_logging()

# Ensure progress directory exists
PROGRESS_DIR = Path("./progress")
PROGRESS_DIR.mkdir(exist_ok=True)


def save_progress(job_id, total, processed, success=True, error=None):
    """
    Saves job progress to a file.
    
    Args:
        job_id (str): Unique job identifier
        total (int): Total number of items to process
        processed (int): Number of items processed so far
        success (bool): Whether the job is successful so far
        error (str, optional): Error message if any
    
    Returns:
        dict: Progress data
    """
    progress_file = PROGRESS_DIR / f"{job_id}_progress.json"
    
    percentage = round((processed / total) * 100 if total > 0 else 0, 2)
    progress_data = {
        "job_id": job_id,
        "total": total,
        "processed": processed,
        "percentage": percentage,
        "success": success,
        "error": error,
        "completed": processed >= total
    }
    
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
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