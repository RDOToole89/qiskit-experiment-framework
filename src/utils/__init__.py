# src/utils/__init__.py

""" Utility functions for quantum experiments """

from .cli_parser import parse_args
from .validation import validate_inputs
from .results import save_results, load_results
from .logger_setup import setup_logger
from .config_loader import load_config


# Maintain the previous structure of ExperimentUtils
class ExperimentUtils:
    """Utility class for quantum experiment management"""

    parse_args = staticmethod(parse_args)
    validate_inputs = staticmethod(validate_inputs)
    save_results = staticmethod(save_results)
    load_results = staticmethod(load_results)
    setup_logger = staticmethod(setup_logger)
    load_config = staticmethod(load_config)


__all__ = [
    "ExperimentUtils",
    "parse_args",
    "validate_inputs",
    "save_results",
    "load_results",
    "setup_logger",
    "load_config",
]
