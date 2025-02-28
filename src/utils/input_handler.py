# src/utils/input_handler.py

from typing import Optional, List, Union, Type
from rich.console import Console
from src.utils.validation import InputValidator


class InputHandler:
    """Handles user input with validation and formatting using rich console."""

    def __init__(self, console: Console, messages: dict):
        """
        Initializes the InputHandler with a rich console and messages dictionary.

        Args:
            console (Console): The rich console instance for rendering prompts.
            messages (dict): The dictionary of message templates.
        """
        self.console = console
        self.messages = messages
        self.validator = InputValidator()

    def get_input(
        self,
        prompt_key: str,
        default: str,
        valid_options: Optional[List[str]] = None,
        valid_options_display: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Gets user input with case-insensitive handling and validation.

        Args:
            prompt_key (str): The key for the prompt message in MESSAGES.
            default (str): Default value if user presses Enter.
            valid_options (list, optional): List of valid options for validation.
            valid_options_display (list, optional): List of options to display in the prompt.
            **kwargs: Additional values to format the prompt message with.

        Returns:
            str: User input or default, normalized to lowercase.
        """
        while True:
            try:
                prompt = self.messages.get(
                    prompt_key,
                    f"[bold red]Missing prompt for key: {prompt_key}[/bold red]",
                )
                format_kwargs = {"default": default}
                if valid_options is not None:
                    format_kwargs["valid_options"] = (
                        valid_options_display
                        if valid_options_display is not None
                        else valid_options
                    )
                format_kwargs.update(kwargs)
                self.console.print(prompt.format(**format_kwargs), end="")
                user_input = input().strip().lower() or default.lower()
                if self.validator.validate_choice(user_input, valid_options):
                    return user_input
                self.console.print(
                    self.messages["invalid_input"].format(
                        input=user_input, options=valid_options
                    )
                )
            except KeyboardInterrupt:
                self.console.print(self.messages["operation_cancelled"])
                return default.lower()

    def get_numeric_input(
        self,
        prompt_key: str,
        default: str,
        expected_type: Type[Union[int, float]] = int,
    ) -> Union[int, float]:
        """
        Prompts the user for a numeric input, handling errors gracefully.

        Args:
            prompt_key (str): The key for the prompt message in MESSAGES.
            default (str): Default value as a string.
            expected_type (type): Expected type (int or float).

        Returns:
            Union[int, float]: The numeric value.

        Raises:
            ValueError: If the input cannot be converted to the expected type.
        """
        while True:
            user_input = self.get_input(prompt_key, default)
            value = self.validator.validate_numeric(user_input, expected_type)
            if value is not None:
                return value
            self.console.print(
                self.messages["invalid_input"].format(
                    input=user_input, options=[expected_type.__name__]
                )
            )
            raise ValueError(
                f"Invalid numeric input: {user_input}. Expected type: {expected_type.__name__}"
            )

    def prompt_yes_no(self, key: str, default: str = "n") -> bool:
        """
        Prompts the user for a yes/no answer.

        Args:
            key (str): The key for the prompt message in MESSAGES.
            default (str): Default value ("y" or "n").

        Returns:
            bool: True if yes, False if no.
        """
        user_input = self.get_input(
            key, default, ["y", "yes", "t", "true", "n", "no", "f", "false"]
        )
        return self.validator.validate_yes_no(user_input)
