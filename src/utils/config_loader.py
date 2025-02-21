# src/quantum_experiment/utils/config_loader.py

import os
import json
import logging
from src.config.paths_config import CONFIG_FILE

logger = logging.getLogger("QuantumExperiment.ConfigLoader")


def load_config(config_file: str = CONFIG_FILE) -> dict:
    """
    Loads experiment configuration from a JSON file.

    Supports noise rates, state params, and custom settings for research (e.g., hypergraphs).

    Args:
        config_file (str): Path to JSON config file (default: CONFIG_FILE).

    Returns:
        dict: Loaded configuration.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        json.JSONDecodeError: If config file is invalid.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"❌ Error: {config_file} not found.")

    with open(config_file, "r") as f:
        config = json.load(f)

    logger.debug(f"✅ Loaded configuration from {config_file}: {config}")
    return config
