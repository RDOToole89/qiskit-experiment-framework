# src/tests/test_params.py

import pytest
import re
from unittest.mock import MagicMock, patch
from src.config.params import apply_defaults, validate_parameters


@pytest.fixture
def base_params():
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


@pytest.fixture
def mock_console():
    return MagicMock()


def test_apply_defaults_missing_params():
    """Test apply_defaults fills in missing parameters correctly."""
    params = {"num_qubits": 4}
    result = apply_defaults(params)
    assert result["num_qubits"] == 4
    assert result["state_type"] == "GHZ"
    assert result["noise_type"] == "DEPOLARIZING"
    assert result["noise_enabled"] is True
    assert result["shots"] == 1024
    assert result["sim_mode"] == "qasm"
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


def test_apply_defaults_overrides():
    """Test apply_defaults respects provided parameters."""
    params = {
        "num_qubits": 2,
        "state_type": "W",
        "noise_type": "BIT_FLIP",
        "noise_enabled": False,
        "shots": 2048,
        "sim_mode": "density",
        "visualization_type": "plot",
        "error_rate": 0.05,
    }
    result = apply_defaults(params)
    assert result["num_qubits"] == 2
    assert result["state_type"] == "W"
    assert result["noise_type"] == "BIT_FLIP"
    assert result["noise_enabled"] is False
    assert result["shots"] == 2048
    assert result["sim_mode"] == "density"
    assert result["visualization_type"] == "plot"
    assert result["error_rate"] == 0.05


def test_validate_parameters_missing_required(base_params):
    """Test validate_parameters raises ValueError for missing required parameters."""
    del base_params["num_qubits"]
    with pytest.raises(
        ValueError, match=re.escape("Missing required parameters: ['num_qubits']")
    ):
        validate_parameters(base_params)


def test_validate_parameters_invalid_num_qubits(base_params):
    """Test validate_parameters raises ValueError for invalid num_qubits."""
    base_params["num_qubits"] = 0
    with pytest.raises(ValueError, match="num_qubits must be an integer >= 1"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_state_type(base_params):
    """Test validate_parameters raises ValueError for invalid state_type."""
    base_params["state_type"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid state type: INVALID"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_noise_type(base_params):
    """Test validate_parameters raises ValueError for invalid noise_type."""
    base_params["noise_type"] = "INVALID"
    with pytest.raises(ValueError, match="Invalid noise type: INVALID"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_shots(base_params):
    """Test validate_parameters raises ValueError for invalid shots."""
    base_params["shots"] = 0
    with pytest.raises(ValueError, match="shots must be an integer >= 1"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_sim_mode(base_params):
    """Test validate_parameters raises ValueError for invalid sim_mode."""
    base_params["sim_mode"] = "invalid"
    with pytest.raises(ValueError, match="sim_mode must be either 'qasm' or 'density'"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_error_rate(base_params):
    """Test validate_parameters raises ValueError for invalid error_rate."""
    base_params["error_rate"] = 1.5
    with pytest.raises(ValueError, match="error_rate must be between 0 and 1"):
        validate_parameters(base_params)


def test_validate_parameters_invalid_phase_flip_probs(base_params, mock_console):
    """Test validate_parameters resets z_prob/i_prob for invalid PHASE_FLIP probabilities."""
    with patch("src.config.params.console.print", mock_console):
        base_params["noise_type"] = "PHASE_FLIP"
        base_params["z_prob"] = 0.7
        base_params["i_prob"] = 0.4  # Does not sum to 1
        result = validate_parameters(base_params)
        assert result["z_prob"] is None
        assert result["i_prob"] is None
        mock_console.assert_called_once_with(
            "[bold red]⚠️ Z and I probabilities must sum to 1 and be between 0 and 1.[/bold red]"
        )


def test_validate_parameters_missing_phase_flip_prob(base_params):
    """Test validate_parameters does not reset z_prob/i_prob if only one is provided."""
    base_params["noise_type"] = "PHASE_FLIP"
    base_params["z_prob"] = 0.5
    base_params["i_prob"] = None
    result = validate_parameters(base_params)
    assert result["z_prob"] == 0.5  # Should not be reset since validation doesn't apply
    assert result["i_prob"] is None


def test_validate_parameters_invalid_thermal_relaxation(base_params, mock_console):
    """Test validate_parameters resets T1/T2 for invalid THERMAL_RELAXATION parameters."""
    with patch("src.config.params.console.print", mock_console):
        base_params["noise_type"] = "THERMAL_RELAXATION"
        base_params["t1"] = 100
        base_params["t2"] = 120  # t2 > t1
        result = validate_parameters(base_params)
        assert result["t1"] is None
        assert result["t2"] is None
        mock_console.assert_called_once_with(
            "[bold red]⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation.[/bold red]"
        )


def test_validate_parameters_missing_thermal_relaxation(base_params):
    """Test validate_parameters does not reset t1/t2 if only one is provided."""
    base_params["noise_type"] = "THERMAL_RELAXATION"
    base_params["t1"] = 100
    base_params["t2"] = None
    result = validate_parameters(base_params)
    assert result["t1"] == 100  # Should not be reset since validation doesn't apply
    assert result["t2"] is None


def test_validate_parameters_noise_warning(base_params, mock_console):
    """Test that validate_parameters issues a warning for 1-qubit noise with multi-qubit systems."""
    with patch("src.config.params.console.print", mock_console):
        base_params["noise_type"] = "BIT_FLIP"
        base_params["num_qubits"] = 2  # Multi-qubit system
        result = validate_parameters(base_params)
        mock_console.assert_called_once_with(
            "[bold yellow]⚠️ Warning: BIT_FLIP noise is designed for single-qubit systems, "
            "but you requested 2 qubits. This noise will only be applied to "
            "single-qubit gates ('id', 'u1', 'u2', 'u3').[/bold yellow]"
        )


def test_validate_parameters_density_noise_warning(base_params, mock_console):
    """Test that validate_parameters disables noise for single-qubit noise in density mode."""
    with patch("src.config.params.console.print", mock_console):
        base_params["sim_mode"] = "density"
        base_params["noise_type"] = "BIT_FLIP"
        base_params["noise_enabled"] = True
        result = validate_parameters(base_params)
        assert result["noise_enabled"] is False
        mock_console.assert_called_once_with(
            "[bold yellow]⚠️ Warning: BIT_FLIP noise only applies to single-qubit gates, which are skipped in density matrix simulation mode. "
            "No noise will be applied with this configuration. Noise will be disabled to proceed. "
            "Consider using multi-qubit noise types (e.g., DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION) for density mode.[/bold yellow]"
        )
