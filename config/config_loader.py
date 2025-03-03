"""
Configuration Loader Module
--------------------------
Handles loading and validation of application configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from .logger_config import get_logger

logger = get_logger('log')


class ConfigurationError(Exception):
    """Raised when there's an error in the configuration."""
    pass


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dict containing the configuration

    Raises:
        ConfigurationError: If configuration is invalid or missing required fields
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate configuration
        validate_config(config)

        # Convert paths to absolute paths
        config = process_paths(config)

        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing configuration file: {str(e)}")
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {str(e)}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration structure and required fields.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_sections = ['paths', 'stations', 'processing', 'logging']
    required_paths = ['input_directory', 'output_directory', 'merged_data', 'reports']
    required_processing = ['merge', 'congruency']

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Check paths
    for path in required_paths:
        if path not in config['paths']:
            raise ConfigurationError(f"Missing required path: {path}")

    # Check stations
    if 'point_names' not in config['stations']:
        raise ConfigurationError("Missing point_names in stations section")
    if not isinstance(config['stations']['point_names'], list):
        raise ConfigurationError("point_names must be a list")
    if not config['stations']['point_names']:
        raise ConfigurationError("point_names list cannot be empty")

    # Check processing settings
    for section in required_processing:
        if section not in config['processing']:
            raise ConfigurationError(f"Missing required processing section: {section}")

    # Validate specific processing settings
    merge_settings = config['processing']['merge']
    if not isinstance(merge_settings.get('dropna', False), bool):
        raise ConfigurationError("merge.dropna must be a boolean")
    if not isinstance(merge_settings.get('fixed_solution_only', False), bool):
        raise ConfigurationError("merge.fixed_solution_only must be a boolean")

    congruency = config['processing']['congruency']
    if not isinstance(congruency.get('use_wls', False), bool):
        raise ConfigurationError("congruency.use_wls must be a boolean")
    if not isinstance(congruency.get('m_coef', 1.0), (int, float)):
        raise ConfigurationError("congruency.m_coef must be a number")
    if not isinstance(congruency.get('max_drop', 2), int):
        raise ConfigurationError("congruency.max_drop must be an integer")


def process_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert relative paths to absolute paths in the configuration.

    Args:
        config: Configuration dictionary with paths

    Returns:
        Updated configuration dictionary with absolute paths
    """
    project_root = Path.cwd()
    paths = config['paths']

    # Convert paths to absolute paths
    for key, path in paths.items():
        if path:  # Skip None values
            abs_path = project_root / path
            paths[key] = str(abs_path)

            # Create directories if they don't exist
            if key != 'merged_data':  # Don't create file paths
                os.makedirs(abs_path, exist_ok=True)

    config['paths'] = paths
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.

    Returns:
        Dict containing default configuration values
    """
    return {
        'paths': {
            'input_directory': '2024-08-30',
            'output_directory': 'Data',
            'merged_data': 'Data/input_files/merged_data.csv',
            'reports': 'Data/reports'
        },
        'stations': {
            'point_names': [
                "SNSK00RUS", "SNSK01RUS", "SNSK02RUS", "SNSK03RUS"
            ]
        },
        'processing': {
            'merge': {
                'resample_interval': None,
                'dropna': False,
                'fixed_solution_only': False
            },
            'congruency': {
                'use_wls': False,
                'window_size': '1h',
                'Q_status': '0',
                'Qdd_status': '0',
                'm_coef': 1.0,
                'max_drop': 2
            }
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'capture_for_report': True
        }
    }
