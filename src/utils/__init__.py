# src/utils/__init__.py

from .cli import parse_args
from .logger import setup_logger
from .validation import validate_inputs
from .results import save_results, load_results
from .config_loader import load_config

class ExperimentUtils:
    parse_args = staticmethod(parse_args)
    setup_logger = staticmethod(setup_logger)
    validate_inputs = staticmethod(validate_inputs)
    save_results = staticmethod(save_results)
    load_results = staticmethod(load_results)
    load_config = staticmethod(load_config)
