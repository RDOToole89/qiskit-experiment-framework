import os
import json
import logging

logger = logging.getLogger("QuantumExperiment.Utils")

def load_config(config_file="config.json") -> dict:
    """
    Loads experiment configuration from a JSON file.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Error: {config_file} not found.")

    with open(config_file, "r") as f:
        config = json.load(f)
    logger.debug(f"Loaded configuration from {config_file}: {config}")
    return config
