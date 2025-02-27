# src/config/params.py

from typing import Dict, Optional
from rich.console import Console
from .defaults import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)
from .constants import VALID_NOISE_TYPES, VALID_STATE_TYPES, NOISE_SHORTCUTS

console = Console()

def validate_parameters(args: Dict) -> Dict:
    """
    Validates experiment parameters and applies defaults.

    Args:
        args (Dict): Experiment parameters.

    Returns:
        Dict: Validated parameters.

    Raises:
        ValueError: If parameters are invalid.
    """
    validated_args = args.copy()

    # Noise type validation
    noise_input = validated_args["noise_type"].lower()
    validated_args["noise_type"] = NOISE_SHORTCUTS.get(noise_input, noise_input.upper())
    if validated_args["noise_type"] not in VALID_NOISE_TYPES:
        console.print(f"[bold red]Error: Invalid noise type '{validated_args['noise_type']}'. Choose from {VALID_NOISE_TYPES}.[/bold red]")
        raise ValueError(f"Invalid noise type: {validated_args['noise_type']}")

    # State type validation
    validated_args["state_type"] = validated_args["state_type"].upper()
    if validated_args["state_type"] not in VALID_STATE_TYPES:
        console.print(f"[bold red]Error: Invalid state type '{validated_args['state_type']}'. Choose from {VALID_STATE_TYPES}.[/bold red]")
        raise ValueError(f"Invalid state type: {validated_args['state_type']}")

    # Validate Z/I probabilities for PHASE_FLIP
    if (
        validated_args["noise_type"] == "PHASE_FLIP"
        and validated_args["z_prob"] is not None
        and validated_args["i_prob"] is not None
    ):
        if not (
            0 <= validated_args["z_prob"] <= 1
            and 0 <= validated_args["i_prob"] <= 1
            and abs(validated_args["z_prob"] + validated_args["i_prob"] - 1) < 1e-10
        ):
            console.print("[bold red]⚠️ Z and I probabilities must sum to 1 and be between 0 and 1.[/bold red]")
            validated_args["z_prob"], validated_args["i_prob"] = None, None

    # Validate T1/T2 for THERMAL_RELAXATION
    if (
        validated_args["noise_type"] == "THERMAL_RELAXATION"
        and validated_args["t1"] is not None
        and validated_args["t2"] is not None
    ):
        if validated_args["t1"] <= 0 or validated_args["t2"] <= 0 or validated_args["t2"] > validated_args["t1"]:
            console.print("[bold red]⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation.[/bold red]")
            validated_args["t1"], validated_args["t2"] = None, None

    # Warn about 1-qubit noise limits
    if (
        validated_args["noise_type"] in ["AMPLITUDE_DAMPING", "PHASE_DAMPING", "BIT_FLIP"]
        and validated_args["num_qubits"] > 1
    ):
        console.print("[bold yellow]⚠️ Note: This noise type applies only to 1-qubit gates, skipping multi-qubit gates (e.g., CNOTs).[/bold yellow]")

    return validated_args

def apply_defaults(args: Dict) -> Dict:
    """
    Applies default values to missing parameters.

    Args:
        args (Dict): Experiment parameters.

    Returns:
        Dict: Parameters with defaults applied.
    """
    defaults = {
        "num_qubits": DEFAULT_NUM_QUBITS,
        "state_type": DEFAULT_STATE_TYPE,
        "noise_type": DEFAULT_NOISE_TYPE,
        "noise_enabled": DEFAULT_NOISE_ENABLED,
        "shots": DEFAULT_SHOTS,
        "sim_mode": DEFAULT_SIM_MODE,
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
    defaults.update(args)
    return defaults
