"""Logging configuration module for the ReedsShepp-MLOps application.

This module provides a centralized logging configuration that writes logs to both
console and a daily rotating log file. It ensures consistent log formatting
across the entire application and handles log file management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Directory where log files will be stored
LOGS_DIR = Path("logs")
# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
# Daily log file name with date stamp
LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"

# Configure root logger with basic settings
logging.basicConfig(
    level=logging.INFO,  # Default log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # Write logs to file with daily rotation
        logging.FileHandler(LOG_FILE, mode="a"),
        # Also output to console
        logging.StreamHandler(),
    ],
)

# Set higher log level for file handler to reduce disk I/O
logging.getLogger().handlers[0].setLevel(logging.INFO)
# Set console handler to show warnings and above by default
logging.getLogger().handlers[1].setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance for the specified module.

    This function provides a consistent way to obtain logger instances throughout
    the application. The logger will inherit the root configuration but can be
    further customized by the calling module.

    Args:
        name: The name of the logger. If None or '__main__', the root logger is returned.
              Typically, modules should pass __name__ to get a namespaced logger.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        >>> # In your module
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an informational message")
        >>> logger.error("This is an error message")
    """
    logger = logging.getLogger(name)
    return logger
