"""
Logging utilities for the MRI preprocessing pipeline.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union


def setup_main_logger(
        log_dir: Union[str, Path],
        log_level: int = logging.INFO,
        console_level: int = None
) -> logging.Logger:
    """
    Set up the main logger for the application.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level for file output
        console_level: Logging level for console output (defaults to log_level)

    Returns:
        Configured logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if console_level is None:
        console_level = log_level

    # Get or create logger
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything

    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"preprocessing_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")

    return logger


def setup_error_logger(log_dir: Union[str, Path]) -> logging.Logger:
    """
    Set up a dedicated error logger that overwrites previous log file.

    Args:
        log_dir: Directory to store error log file

    Returns:
        Configured error logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get or create logger
    error_logger = logging.getLogger("preprocessing.errors")
    error_logger.setLevel(logging.ERROR)

    # Clear existing handlers
    if error_logger.handlers:
        error_logger.handlers.clear()

    # Create file handler with 'w' mode to overwrite previous log
    error_file = log_dir / "errors.log"
    file_handler = logging.FileHandler(error_file, mode='w')
    file_handler.setLevel(logging.ERROR)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    error_logger.addHandler(file_handler)

    return error_logger


def get_logger(name: str, debug: bool = False) -> logging.Logger:
    """
    Get a module-specific logger.

    Args:
        name: Logger name/identifier
        debug: Whether to enable debug logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"preprocessing.{name}")

    # Set level based on debug flag
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    return logger