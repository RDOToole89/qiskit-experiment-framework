# src/config/logging_config.py

"""
📌 Logging Configuration

Manages logging behavior for quantum experiments.
"""

import logging

DEFAULT_LOG_LEVEL = "INFO"  # Options: ["DEBUG", "INFO", "WARNING", "ERROR"]


def setup_logging():
    logging.basicConfig(
        level=DEFAULT_LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/experiment.log"),
            logging.StreamHandler(),
        ],
    )


logger = logging.getLogger("QuantumExperiment")
