#!/usr/bin/env python3
"""
AnyDataset - Startup Script

This script handles setup and launching of the AnyDataset application.
- Creates required directories
- Sets proper permissions
- Starts the FastAPI server
"""
import os
import sys
import shutil
from pathlib import Path
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AnyDataset-Setup")

def prepare_environment():
    """Create necessary directories and check permissions."""
    # Get application directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    app_dir = script_dir / "app"

    # Required directories
    required_dirs = [
        app_dir / "uploads",
        app_dir / "ready",
        app_dir / "progress",
        app_dir / "logs"
    ]

    # Create directories if they don't exist
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Directory ready: {dir_path}")

            # Check write permissions by creating and removing a test file
            test_file = dir_path / ".permission_test"
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()  # Remove test file
            except Exception as e:
                logger.warning(f"Directory {dir_path} is not writable: {e}")
                logger.warning(f"Trying to set permissions...")

                # Try to set permissions (this will work on UNIX systems)
                try:
                    os.chmod(dir_path, 0o755)
                    logger.info(f"Permissions updated for {dir_path}")
                except Exception as perm_error:
                    logger.error(f"Failed to set permissions: {perm_error}")
                    logger.error(f"Please manually ensure {dir_path} is writable")

        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            logger.error("Application may not function correctly. Please check permissions.")

def find_executable(name):
    """Find an executable in PATH."""
    return shutil.which(name)

def start_application():
    """Start the FastAPI application."""
    app_path = os.path.join("app", "app.py")
    
    # Check for available launcher methods
    uv_path = find_executable("uv")
    uvicorn_path = find_executable("uvicorn")
    python_path = find_executable("python") or find_executable("python3")
    
    logger.info("Starting AnyDataset application...")
    
    # Use uv if available (fastest)
    if uv_path:
        logger.info("Starting with uv (recommended)")
        return subprocess.run([uv_path, "run", app_path])
    
    # Use uvicorn directly if available
    elif uvicorn_path:
        logger.info("Starting with uvicorn")
        return subprocess.run([uvicorn_path, "app.app:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Fall back to python
    elif python_path:
        logger.info("Starting with python")
        return subprocess.run([python_path, app_path])
    
    else:
        logger.error("No suitable Python runtime found. Please install Python 3.7+")
        return 1

if __name__ == "__main__":
    try:
        # Ensure we're in the right directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Prepare environment
        prepare_environment()
        
        # Start application
        result = start_application()
        sys.exit(result.returncode if hasattr(result, 'returncode') else 0)
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)