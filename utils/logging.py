"""
Logging utilities.
"""

import os
import logging
from pathlib import Path


def setup_logging(log_level=None):
    """
    Configures the logging system.
    
    Args:
        log_level (str, optional): Log level to use. If None, uses LOG_LEVEL from env
        
    Returns:
        Logger: Configured logger
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "anydataset.log")
        ]
    )
    
    return logging.getLogger("anydataset")