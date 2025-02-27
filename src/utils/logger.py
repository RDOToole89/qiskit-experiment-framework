# src/utils/logger.py

import os
import logging
import json
from datetime import datetime
from typing import Optional

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that supports both human-readable logs and structured JSON for analysis.
    """
    def format(self, record):
        # Basic log message with timestamp, name, level, and message
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S,%f"),
            "name": record.name,
            "level": record.levelname,
            "message": record.msg % record.args if record.args else record.msg,
            "filename": record.filename,
            "lineno": record.lineno,
            "experiment_id": getattr(record, "experiment_id", "N/A"),
        }
        # Add extra fields if provided (e.g., circuit details, metrics)
        if hasattr(record, "extra_info"):
            log_entry["extra_info"] = record.extra_info

        # For console: human-readable format
        if record.levelname in ["INFO", "WARNING"]:
            return (
                f"{log_entry['timestamp']} - {log_entry['name']} - {log_entry['level']} - "
                f"{log_entry['message']} (Experiment ID: {log_entry['experiment_id']})"
            )
        # For DEBUG or ERROR: include file and line number
        elif record.levelname in ["DEBUG", "ERROR"]:
            return (
                f"{log_entry['timestamp']} - {log_entry['name']} - {log_entry['level']} - "
                f"{log_entry['message']} (Experiment ID: {log_entry['experiment_id']}) "
                f"[{log_entry['filename']}:{log_entry['lineno']}]"
            )
        # For structured logging (e.g., to a JSON file)
        return json.dumps(log_entry)

def setup_logger(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    structured_log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configures logging with support for file, console, and structured JSON output.

    Args:
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR").
        log_to_file (bool): Whether to log to a file.
        log_to_console (bool): Whether to log to the console.
        structured_log_file (str, optional): Path to a JSON file for structured logs.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    # Create logger
    logger = logging.getLogger("QuantumExperiment")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()  # Clear any existing handlers

    # Formatter
    formatter = StructuredFormatter()

    # File handler (human-readable)
    handlers = []
    if log_to_file:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    # Structured JSON handler (optional)
    if structured_log_file:
        structured_handler = logging.FileHandler(structured_log_file)
        structured_handler.setFormatter(formatter)
        handlers.append(structured_handler)

    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)

    # Initial log to confirm setup
    logger.debug("Logger configured for quantum experiment utilities", extra={"experiment_id": "N/A"})
    return logger

# Helper to attach experiment ID to log records
def log_with_experiment_id(logger: logging.Logger, level: str, message: str, experiment_id: str, extra_info: Optional[dict] = None):
    """
    Logs a message with an experiment ID and optional extra info.

    Args:
        logger: Logger instance.
        level: Log level ("debug", "info", "warning", "error").
        message: Log message.
        experiment_id: Unique identifier for the experiment.
        extra_info: Optional dictionary with additional metadata.
    """
    extra = {"experiment_id": experiment_id}
    if extra_info:
        extra["extra_info"] = extra_info
    getattr(logger, level.lower())(message, extra=extra)
