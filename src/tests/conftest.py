# tests/conftest.py

import pytest
from unittest.mock import MagicMock
from rich.console import Console

@pytest.fixture
def mock_console(monkeypatch):
    """
    Fixture to mock rich.console.Console for capturing output.
    """
    mock = MagicMock(spec=Console)
    monkeypatch.setattr("src.config.params.console", mock)
    return mock

@pytest.fixture
def base_params():
    """
    Fixture to provide a base set of parameters for testing.
    """
    return {
        "num_qubits": 3,
        "state_type": "GHZ",
        "noise_type": "DEPOLARIZING",
        "noise_enabled": True,
        "shots": 1024,
        "sim_mode": "qasm",
        "visualization_type": "none",
        "save_plot": None,
        "min_occurrences": 0,
        "show_real": False,
        "show_imag": False,
        "error_rate": None,
        "z_prob": None,
        "i_prob": None,
        "t1": None,
        "t2": None,
        "custom_params": None,
    }
