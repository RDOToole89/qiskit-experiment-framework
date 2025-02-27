# tests/test_params.py

import pytest
from src.config.params import apply_defaults, validate_parameters
from src.config.constants import VALID_NOISE_TYPES, VALID_STATE_TYPES, NOISE_SHORTCUTS
from src.config.defaults import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)

def test_apply_defaults_empty_input():
    """Test that apply_defaults fills in all parameters with defaults when input is empty."""
    args = {}
    result = apply_defaults(args)
    assert result["num_qubits"] == DEFAULT_NUM_QUBITS
    assert result["state_type"] == DEFAULT_STATE_TYPE
    assert result["noise_type"] == DEFAULT_NOISE_TYPE
    assert result["noise_enabled"] == DEFAULT_NOISE_ENABLED
    assert result["shots"] == DEFAULT_SHOTS
    assert result["sim_mode"] == DEFAULT_SIM_MODE
    assert result["visualization_type"] == "none"
    assert result["save_plot"] is None
    assert result["min_occurrences"] == 0
    assert result["show_real"] is False
    assert result["show_imag"] is False
    assert result["error_rate"] is None
    assert result["z_prob"] is None
    assert result["i_prob"] is None
    assert result["t1"] is None
    assert result["t2"] is None
    assert result["custom_params"] is None

def test_apply_defaults_partial_input():
    """Test that apply_defaults preserves provided values and fills in missing ones."""
    args = {
        "num_qubits": 4,
        "state_type": "W",
        "error_rate": 0.2
    }
    result = apply_defaults(args)
    assert result["num_qubits"] == 4  # Provided value
    assert result["state_type"] == "W"  # Provided value
    assert result["error_rate"] == 0.2  # Provided value
    assert result["noise_type"] == DEFAULT_NOISE_TYPE  # Default
    assert result["noise_enabled"] == DEFAULT_NOISE_ENABLED  # Default
    assert result["shots"] == DEFAULT_SHOTS  # Default

def test_validate_parameters_valid_input(base_params):
    """Test that validate_parameters accepts valid inputs."""
    result = validate_parameters(base_params)
    assert result["noise_type"] == "DEPOLARIZING"
    assert result["state_type"] == "GHZ"
    assert result["num_qubits"] == 3

def test_validate_parameters_invalid_noise_type(base_params, mock_console):
    """Test that validate_parameters rejects invalid noise types."""
    base_params["noise_type"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid noise type: INVALID"):
        validate_parameters(base_params)
    mock_console.print.assert_called_once_with(
        f"[bold red]Error: Invalid noise type 'INVALID'. Choose from {VALID_NOISE_TYPES}.[/bold red]"
    )

def test_validate_parameters_invalid_state_type(base_params, mock_console):
    """Test that validate_parameters rejects invalid state types."""
    base_params["state_type"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid state type: INVALID"):
        validate_parameters(base_params)
    mock_console.print.assert_called_once_with(
        f"[bold red]Error: Invalid state type 'INVALID'. Choose from {VALID_STATE_TYPES}.[/bold red]"
    )

def test_validate_parameters_noise_shortcut(base_params):
    """Test that validate_parameters handles noise shortcuts correctly."""
    base_params["noise_type"] = "d"  # Shortcut for DEPOLARIZING
    result = validate_parameters(base_params)
    assert result["noise_type"] == "DEPOLARIZING"

def test_validate_parameters_phase_flip_invalid_probs(base_params, mock_console):
    """Test that validate_parameters resets Z/I probabilities if they don't sum to 1."""
    base_params["noise_type"] = "PHASE_FLIP"
    base_params["z_prob"] = 0.7
    base_params["i_prob"] = 0.5  # Sum = 1.2, invalid
    result = validate_parameters(base_params)
    assert result["z_prob"] is None
    assert result["i_prob"] is None
    mock_console.print.assert_called_once_with(
        "[bold red]⚠️ Z and I probabilities must sum to 1 and be between 0 and 1.[/bold red]"
    )

def test_validate_parameters_phase_flip_valid_probs(base_params):
    """Test that validate_parameters accepts valid Z/I probabilities."""
    base_params["noise_type"] = "PHASE_FLIP"
    base_params["z_prob"] = 0.4
    base_params["i_prob"] = 0.6  # Sum = 1.0, valid
    result = validate_parameters(base_params)
    assert result["z_prob"] == 0.4
    assert result["i_prob"] == 0.6

def test_validate_parameters_thermal_relaxation_invalid_t1_t2(base_params, mock_console):
    """Test that validate_parameters resets T1/T2 if T1 <= 0, T2 <= 0, or T2 > T1."""
    base_params["noise_type"] = "THERMAL_RELAXATION"
    base_params["t1"] = -1  # Invalid: T1 <= 0
    base_params["t2"] = 0.5
    result = validate_parameters(base_params)
    assert result["t1"] is None
    assert result["t2"] is None
    mock_console.print.assert_called_once_with(
        "[bold red]⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation.[/bold red]"
    )

def test_validate_parameters_thermal_relaxation_t2_greater_than_t1(base_params, mock_console):
    """Test that validate_parameters resets T1/T2 if T2 > T1."""
    base_params["noise_type"] = "THERMAL_RELAXATION"
    base_params["t1"] = 0.1
    base_params["t2"] = 0.2  # Invalid: T2 > T1
    result = validate_parameters(base_params)
    assert result["t1"] is None
    assert result["t2"] is None
    mock_console.print.assert_called_once_with(
        "[bold red]⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation.[/bold red]"
    )

def test_validate_parameters_thermal_relaxation_valid_t1_t2(base_params):
    """Test that validate_parameters accepts valid T1/T2 values."""
    base_params["noise_type"] = "THERMAL_RELAXATION"
    base_params["t1"] = 0.2
    base_params["t2"] = 0.1  # Valid: T2 <= T1
    result = validate_parameters(base_params)
    assert result["t1"] == 0.2
    assert result["t2"] == 0.1

def test_validate_parameters_noise_warning(base_params, mock_console):
    """Test that validate_parameters issues a warning for 1-qubit noise with multi-qubit systems."""
    base_params["noise_type"] = "BIT_FLIP"
    base_params["num_qubits"] = 2  # Multi-qubit system
    result = validate_parameters(base_params)
    mock_console.print.assert_called_once_with(
        "[bold yellow]⚠️ Note: This noise type applies only to 1-qubit gates, skipping multi-qubit gates (e.g., CNOTs).[/bold yellow]"
    )
