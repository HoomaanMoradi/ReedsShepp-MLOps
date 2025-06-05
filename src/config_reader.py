"""Configuration file reader for the ReedsShepp-MLOps project.

This module provides functionality to read and parse YAML configuration files
used throughout the application. It handles file operations, parsing, and
basic validation of configuration files.

Example:
    >>> from config_reader import read_config
    >>> config = read_config("config/config.yaml")
    >>> model_config = config["model_training"]
"""

import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def read_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a YAML configuration file with comprehensive error handling.

    This function loads a YAML configuration file, validates its existence and
    accessibility, parses its contents, and returns the configuration as a dictionary.
    It includes detailed error handling and logging for better debugging.

    Args:
        config_path: Path to the YAML configuration file. Can be a string or Path object.
                    Can be either an absolute path or relative to the current working directory.

    Returns:
        A dictionary containing the parsed YAML configuration.
        The structure of the dictionary corresponds to the YAML file structure.

    Raises:
        FileNotFoundError: If the config file does not exist or is not accessible.
        PermissionError: If the config file cannot be read due to permission issues.
        yaml.YAMLError: If there is an error parsing the YAML content.
        OSError: For other I/O related errors.

    Example:
        >>> config = read_config("config/config.yaml")
        >>> model_config = config["model_training"]
        >>> data_config = config["data_ingestion"]

    Note:
        - The function uses yaml.safe_load() for security (avoids arbitrary code execution).
        - All file paths in the config should be relative to the config file's directory.
    """
    try:
        # Convert to Path object if it's a string
        config_file = Path(config_path).resolve()

        # Check if file exists and is accessible
        if not config_file.exists():
            error_msg = f"Configuration file not found at: {config_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not config_file.is_file():
            error_msg = f"Configuration path is not a file: {config_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Read and parse the YAML file
        try:
            with open(config_file, "r", encoding="utf-8") as file:
                logger.info(f"Loading configuration from: {config_file}")
                config = yaml.safe_load(file)

                if config is None:
                    logger.warning("Configuration file is empty")
                    return {}

                if not isinstance(config, dict):
                    error_msg = "Configuration file must contain a dictionary/mapping"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.debug(
                    f"Successfully loaded configuration with keys: {list(config.keys())}"
                )
                return config

        except yaml.YAMLError as e:
            error_msg = f"Error parsing YAML file {config_file}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise yaml.YAMLError(error_msg) from e

    except (OSError, PermissionError) as e:
        error_msg = f"Error accessing configuration file {config_file}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
