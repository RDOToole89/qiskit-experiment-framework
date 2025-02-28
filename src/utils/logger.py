# src/utils/logger.py

import os
import logging
import json
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.text import Text

console = Console()


class RichHandler(logging.Handler):
    """
    A custom logging handler that uses rich to render console output with markup.
    """

    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)
        self.console = console

    def emit(self, record):
        try:
            msg = self.format(record)
            # Render the message using rich markup
            text = Text.from_markup(msg)
            self.console.print(text)
        except Exception:
            self.handleError(record)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that supports both human-readable logs and structured JSON for analysis.
    Formats messages with rich markup for console output.
    """

    def __init__(self, is_rich_handler: bool = False):
        super().__init__()
        self.is_rich_handler = is_rich_handler

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

        # Define color coding based on log level
        level_colors = {
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "DEBUG": "cyan",
        }
        level_color = level_colors.get(record.levelname, "white")

        # For RichHandler (console): return rich markup
        if self.is_rich_handler:
            if record.levelname in ["INFO", "WARNING"]:
                message = (
                    f"[bold {level_color}]{log_entry['timestamp']} - {log_entry['name']} - "
                    f"{log_entry['level']} - {log_entry['message']} "
                    f"(Experiment ID: {log_entry['experiment_id']})[/bold {level_color}]"
                )
                if hasattr(record, "extra_info"):
                    message += (
                        f" [dim](Extra: {json.dumps(log_entry['extra_info'])})[/dim]"
                    )
                return message
            elif record.levelname in ["DEBUG", "ERROR"]:
                message = (
                    f"[bold {level_color}]{log_entry['timestamp']} - {log_entry['name']} - "
                    f"{log_entry['level']} - {log_entry['message']} "
                    f"(Experiment ID: {log_entry['experiment_id']}) "
                    f"[{log_entry['filename']}:{log_entry['lineno']}][/bold {level_color}]"
                )
                if hasattr(record, "extra_info"):
                    message += (
                        f" [dim](Extra: {json.dumps(log_entry['extra_info'])})[/dim]"
                    )
                return message
        # For file or structured JSON logging: return JSON
        return json.dumps(log_entry)


def setup_logger(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    structured_log_file: Optional[str] = None,
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

    # Ensure timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{timestamp}_experiment.log")

    # Create logger
    logger = logging.getLogger("QuantumExperiment")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()  # Clear any existing handlers

    # File handler (human-readable)
    handlers = []
    if log_to_file:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(StructuredFormatter(is_rich_handler=False))
        handlers.append(file_handler)

    # Console handler with rich output
    if log_to_console:
        console_handler = RichHandler()
        console_handler.setFormatter(StructuredFormatter(is_rich_handler=True))
        handlers.append(console_handler)

    # Structured JSON handler (optional)
    if structured_log_file:
        structured_handler = logging.FileHandler(structured_log_file)
        structured_handler.setFormatter(StructuredFormatter(is_rich_handler=False))
        handlers.append(structured_handler)

    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)

    # Initial log to confirm setup
    logger.debug(
        "Logger configured for quantum experiment utilities",
        extra={"experiment_id": "N/A"},
    )
    return logger


# Helper to attach experiment ID to log records
def log_with_experiment_id(
    logger: logging.Logger,
    level: str,
    message: str,
    experiment_id: str,
    extra_info: Optional[dict] = None,
):
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
