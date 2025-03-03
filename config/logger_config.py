"""
Logger Configuration Module
Provides centralized logging configuration for the entire project.
"""

import logging
from io import StringIO
from typing import Optional


def setup_logger(name: Optional[str] = None,
                 level: int = logging.INFO,
                 capture_string: bool = False) -> tuple[logging.Logger, Optional[StringIO]]:
    """
    Configure and return a logger instance with optional string capture capability.

    Args:
        name: Logger name (if None, returns root logger)
        level: Logging level (default: INFO)
        capture_string: Whether to capture logs in a StringIO buffer

    Returns:
        tuple: (logger instance, StringIO handler if capture_string=True else None)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create StringIO handler if requested
    string_io_handler = None
    if capture_string:
        string_io = StringIO()
        string_io_handler = logging.StreamHandler(string_io)
        string_io_handler.setLevel(level)
        string_io_handler.setFormatter(formatter)
        logger.addHandler(string_io_handler)

    return logger, string_io_handler


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If it doesn't exist, creates one with default configuration.

    Args:
        name: Logger name (if None, returns root logger)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger, _ = setup_logger(name)
    return logger
