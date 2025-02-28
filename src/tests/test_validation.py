# tests/test_validation.py

import pytest
from src.utils.validation import InputValidator, validate_inputs


def test_validate_choice_valid():
    """
    Test validate_choice with a valid choice.
    """
    validator = InputValidator()
    assert validator.validate_choice("s", ["s", "n", "q"]) is True
    assert validator.validate_choice("S", ["s", "n", "q"]) is True  # Case-insensitive


def test_validate_choice_invalid():
    """
    Test validate_choice with an invalid choice.
    """
    validator = InputValidator()
    assert validator.validate_choice("x", ["s", "n", "q"]) is False


def test_validate_choice_case_sensitive():
    """
    Test validate_choice with case-sensitive comparison.
    """
    validator = InputValidator()
    assert validator.validate_choice("S", ["s", "n", "q"], case_sensitive=True) is False
    assert validator.validate_choice("s", ["s", "n", "q"], case_sensitive=True) is True


def test_validate_choice_no_options():
    """
    Test validate_choice with no valid options.
    """
    validator = InputValidator()
    assert validator.validate_choice("anything", None) is True


def test_validate_numeric_valid_int():
    """
    Test validate_numeric with a valid integer.
    """
    validator = InputValidator()
    assert validator.validate_numeric("42", int) == 42


def test_validate_numeric_valid_float():
    """
    Test validate_numeric with a valid float.
    """
    validator = InputValidator()
    assert validator.validate_numeric("3.14", float) == 3.14


def test_validate_numeric_invalid():
    """
    Test validate_numeric with an invalid numeric input.
    """
    validator = InputValidator()
    assert validator.validate_numeric("invalid", int) is None


def test_validate_yes_no_yes():
    """
    Test validate_yes_no with yes inputs.
    """
    validator = InputValidator()
    assert validator.validate_yes_no("y") is True
    assert validator.validate_yes_no("yes") is True
    assert validator.validate_yes_no("t") is True
    assert validator.validate_yes_no("true") is True


def test_validate_yes_no_no():
    """
    Test validate_yes_no with no inputs.
    """
    validator = InputValidator()
    assert validator.validate_yes_no("n") is False
    assert validator.validate_yes_no("no") is False
    assert validator.validate_yes_no("f") is False
    assert validator.validate_yes_no("false") is False
    assert validator.validate_yes_no("invalid") is False


def test_validate_inputs_valid():
    """
    Test validate_inputs with valid inputs.
    """
    validate_inputs(
        num_qubits=3,
        state_type="GHZ",
        noise_type="DEPOLARIZING",
        sim_mode="qasm",
        error_rate=0.1,
    )  # Should not raise any exception


def test_validate_inputs_invalid_num_qubits():
    """
    Test validate_inputs with invalid number of qubits.
    """
    with pytest.raises(ValueError, match="Number of qubits must be at least 1"):
        validate_inputs(
            num_qubits=0,
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            sim_mode="qasm",
        )


def test_validate_inputs_invalid_state_type():
    """
    Test validate_inputs with invalid state type.
    """
    with pytest.raises(ValueError, match="Invalid state type"):
        validate_inputs(
            num_qubits=3,
            state_type="INVALID",
            noise_type="DEPOLARIZING",
            sim_mode="qasm",
        )


def test_validate_inputs_invalid_noise_type():
    """
    Test validate_inputs with invalid noise type.
    """
    with pytest.raises(ValueError, match="Invalid noise type"):
        validate_inputs(
            num_qubits=3,
            state_type="GHZ",
            noise_type="INVALID",
            sim_mode="qasm",
        )


def test_validate_inputs_invalid_sim_mode():
    """
    Test validate_inputs with invalid simulation mode.
    """
    with pytest.raises(ValueError, match="Invalid simulation mode"):
        validate_inputs(
            num_qubits=3,
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            sim_mode="invalid",
        )
