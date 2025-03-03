"""
Configuration package initialization.
"""
import sys
from pathlib import Path

# Add project root to sys.path for consistent imports
root_path = Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from .logger_config import get_logger, setup_logger
from .config_loader import load_config, ConfigurationError, get_default_config

__all__ = ['get_logger', 'setup_logger', 'load_config', 'ConfigurationError', 'get_default_config']
