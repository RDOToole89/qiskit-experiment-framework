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
from .constants import (
    VALID_NOISE_TYPES,
    VALID_STATE_TYPES,
    NOISE_SHORTCUTS,
    SINGLE_QUBIT_NOISE_TYPES,
)

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

    # Check for missing required parameters
    required = ["num_qubits", "state_type", "noise_type", "shots", "sim_mode"]
    missing = [
        key
        for key in required
        if key not in validated_args or validated_args[key] is None
    ]
    if missing:
        console.print(
            f"[bold red]Error: Missing required parameters: {missing}[/bold red]"
        )
        raise ValueError(f"Missing required parameters: {missing}")

    # Validate num_qubits
    if (
        not isinstance(validated_args["num_qubits"], int)
        or validated_args["num_qubits"] < 1
    ):
        console.print("[bold red]Error: num_qubits must be an integer >= 1[/bold red]")
        raise ValueError("num_qubits must be an integer >= 1")

    # Validate shots
    if not isinstance(validated_args["shots"], int) or validated_args["shots"] < 1:
        console.print("[bold red]Error: shots must be an integer >= 1[/bold red]")
        raise ValueError("shots must be an integer >= 1")

    # Validate sim_mode
    if validated_args["sim_mode"] not in ["qasm", "density"]:
        console.print(
            "[bold red]Error: sim_mode must be either 'qasm' or 'density'[/bold red]"
        )
        raise ValueError("sim_mode must be either 'qasm' or 'density'")

    # Noise type validation
    noise_input = validated_args["noise_type"].lower()
    validated_args["noise_type"] = NOISE_SHORTCUTS.get(noise_input, noise_input.upper())
    if validated_args["noise_type"] not in VALID_NOISE_TYPES:
        console.print(
            f"[bold red]Error: Invalid noise type '{validated_args['noise_type']}'. Choose from {VALID_NOISE_TYPES}.[/bold red]"
        )
        raise ValueError(f"Invalid noise type: {validated_args['noise_type']}")

    # State type validation
    validated_args["state_type"] = validated_args["state_type"].upper()
    if validated_args["state_type"] not in VALID_STATE_TYPES:
        console.print(
            f"[bold red]Error: Invalid state type '{validated_args['state_type']}'. Choose from {VALID_STATE_TYPES}.[/bold red]"
        )
        raise ValueError(f"Invalid state type: {validated_args['state_type']}")

    # Validate error_rate
    if validated_args["error_rate"] is not None:
        if not (0 <= validated_args["error_rate"] <= 1):
            console.print(
                "[bold red]Error: error_rate must be between 0 and 1[/bold red]"
            )
            raise ValueError("error_rate must be between 0 and 1")

    # Validate compatibility of noise type with density matrix simulation
    single_qubit_noise_types = ["AMPLITUDE_DAMPING", "PHASE_DAMPING", "BIT_FLIP"]
    density_noise_warning = False
    if (
        validated_args["sim_mode"] == "density"
        and validated_args["noise_type"] in single_qubit_noise_types
        and validated_args["noise_enabled"]
    ):
        console.print(
            f"[bold yellow]⚠️ Warning: {validated_args['noise_type']} noise only applies to single-qubit gates, which are skipped in density matrix simulation mode. "
            "No noise will be applied with this configuration. Noise will be disabled to proceed. "
            "Consider using multi-qubit noise types (e.g., DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION) for density mode.[/bold yellow]"
        )
        validated_args["noise_enabled"] = False
        density_noise_warning = True

    # Warn about single-qubit noise with multi-qubit systems, but skip if density simulation warning applies
    if not density_noise_warning:
        if (
            validated_args["noise_type"] in SINGLE_QUBIT_NOISE_TYPES
            and validated_args["num_qubits"] > 1
        ):
            console.print(
                f"[bold yellow]⚠️ Warning: {validated_args['noise_type']} noise is designed for single-qubit systems, "
                f"but you requested {validated_args['num_qubits']} qubits. This noise will only be applied to "
                f"single-qubit gates ('id', 'u1', 'u2', 'u3').[/bold yellow]"
            )

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
            console.print(
                "[bold red]⚠️ Z and I probabilities must sum to 1 and be between 0 and 1.[/bold red]"
            )
            validated_args["z_prob"], validated_args["i_prob"] = None, None

    # Validate T1/T2 for THERMAL_RELAXATION
    if (
        validated_args["noise_type"] == "THERMAL_RELAXATION"
        and validated_args["t1"] is not None
        and validated_args["t2"] is not None
    ):
        if (
            validated_args["t1"] <= 0
            or validated_args["t2"] <= 0
            or validated_args["t2"] > validated_args["t1"]
        ):
            console.print(
                "[bold red]⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation.[/bold red]"
            )
            validated_args["t1"], validated_args["t2"] = None, None

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
