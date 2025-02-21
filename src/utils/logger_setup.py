# src/quantum_experiment/utils/logger_setup.py

import logging
from datetime import datetime
from src.config.paths_config import LOG_DIR
import os


def setup_logger() -> logging.Logger:
    """Configures logging with automatic log file creation."""

    # Create timestamped log file
    log_filename = os.path.join(
        LOG_DIR, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Print logs to console
        ],
    )

    logger = logging.getLogger("QuantumExperiment")
    return logger


# Initialize global logger instance
logger = setup_logger()
