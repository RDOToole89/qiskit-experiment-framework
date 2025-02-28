# tests/test_input_handler.py

import pytest
from unittest.mock import patch, Mock
from rich.console import Console
from src.utils.input_handler import InputHandler
from src.utils.messages import MESSAGES


@pytest.fixture
def input_handler():
    """
    Fixture to create an InputHandler instance with a mocked console.
    """
    console = Mock(spec=Console)
    return InputHandler(console, MESSAGES)


def test_get_input_valid_choice(input_handler):
    """
    Test get_input with a valid choice.
    """
    with patch("builtins.input", return_value="s"):
        result = input_handler.get_input("your_choice", "s", ["s", "n", "q"])
        assert result == "s"
        input_handler.console.print.assert_called()


def test_get_input_with_default(input_handler):
    """
    Test get_input returns default value when input is empty.
    """
    with patch("builtins.input", return_value=""):
        result = input_handler.get_input("your_choice", "s", ["s", "n", "q"])
        assert result == "s"
        input_handler.console.print.assert_called()


def test_get_input_invalid_choice(input_handler):
    """
    Test get_input loops on invalid choice and displays error message.
    """
    with patch("builtins.input", side_effect=["x", "s"]):
        result = input_handler.get_input("your_choice", "s", ["s", "n", "q"])
        assert result == "s"
        assert input_handler.console.print.call_count >= 2  # Prompt + error message


def test_get_input_keyboard_interrupt(input_handler):
    """
    Test get_input handles KeyboardInterrupt and returns default.
    """
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = input_handler.get_input("your_choice", "s", ["s", "n", "q"])
        assert result == "s"
        input_handler.console.print.assert_any_call(
            input_handler.messages["operation_cancelled"]
        )


def test_get_numeric_input_valid(input_handler):
    """
    Test get_numeric_input with a valid numeric input.
    """
    with patch("builtins.input", return_value="3"):
        result = input_handler.get_numeric_input("num_qubits_prompt", "3", int)
        assert result == 3
        input_handler.console.print.assert_called()


def test_get_numeric_input_invalid_raises_value_error(input_handler):
    """
    Test get_numeric_input raises ValueError on invalid input.
    """
    with patch("builtins.input", return_value="invalid"):
        with pytest.raises(ValueError, match="Invalid numeric input: invalid"):
            input_handler.get_numeric_input("num_qubits_prompt", "3", int)
        input_handler.console.print.assert_called()


def test_prompt_yes_no_yes(input_handler):
    """
    Test prompt_yes_no returns True for yes input.
    """
    with patch("builtins.input", return_value="y"):
        result = input_handler.prompt_yes_no("custom_error_rate_prompt", "n")
        assert result is True
        input_handler.console.print.assert_called()


def test_prompt_yes_no_no(input_handler):
    """
    Test prompt_yes_no returns False for no input.
    """
    with patch("builtins.input", return_value="n"):
        result = input_handler.prompt_yes_no("custom_error_rate_prompt", "n")
        assert result is False
        input_handler.console.print.assert_called()


def test_prompt_yes_no_default(input_handler):
    """
    Test prompt_yes_no returns default value when input is empty.
    """
    with patch("builtins.input", return_value=""):
        result = input_handler.prompt_yes_no("custom_error_rate_prompt", "n")
        assert result is False  # Default "n" maps to False
        input_handler.console.print.assert_called()
