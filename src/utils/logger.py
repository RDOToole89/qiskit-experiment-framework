import os
import logging
from datetime import datetime

def setup_logger() -> logging.Logger:
    """
    Configures logging with automatic log file creation.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("QuantumExperiment")
    logger.debug("Logger configured for quantum experiment utilities")
    return logger
