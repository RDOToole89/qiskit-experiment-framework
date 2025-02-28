#!/usr/bin/env python3
# src/main.py

"""
Interactive script to run quantum experiments, designed for extensibility and research integration.

This script provides an interactive CLI to execute quantum experiments, supporting:
- Dynamic, case-insensitive parameter selection for states, noise models, and simulation modes.
- Logging and structured results saving for hypergraph or decoherence analysis.
- Optional, non-blocking visualization via histograms, density matrices, or hypergraphs,
  closable with Ctrl+C, with save and filtering options.
- Support for time-stepped simulations to analyze decoherence dynamics.
- Extensible architecture for future quantum state/noise additions and research features,
  with full rerun support.

Example usage:
    python main.py  # Runs interactive mode
    python main.py --num-qubits 3 --state-type GHZ --sim-mode density  # Non-interactive mode
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import matplotlib.pyplot as plt
import numpy as np
import json
import warnings
import uuid
from datetime import datetime
from typing import Optional, Dict, Tuple, Union
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from src.run_experiment import run_experiment
from src.utils import logger, results as ExperimentUtils
from src.utils.input_handler import InputHandler
from src.visualization.visualization_handler import handle_visualization
from src.config.params import apply_defaults, validate_parameters
from src.config.constants import (
    VALID_NOISE_TYPES,
    VALID_STATE_TYPES,
    NOISE_SHORTCUTS,
    SINGLE_QUBIT_NOISE_TYPES,
)
from src.config.defaults import DEFAULT_ERROR_RATE
from src.noise_models.noise_factory import NOISE_CLASSES
from src.utils.messages import MESSAGES  # Import the messages lookup table

# Suppress Qiskit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logger with structured output
logger_instance = logger.setup_logger(
    log_level="INFO",
    log_to_file=True,
    log_to_console=True,
    structured_log_file="logs/structured_logs.json",
)

# Initialize Rich console and input handler
console = Console()
input_handler = InputHandler(console, MESSAGES)


def print_message(key: str, **kwargs) -> None:
    """
    Prints a console message from the MESSAGES lookup table, formatting it with provided kwargs.

    Args:
        key (str): The key to look up the message in MESSAGES.
        **kwargs: Values to format the message with (e.g., noise_type, num_qubits).
    """
    message = MESSAGES.get(key, f"[bold red]Missing prompt for key: {key}[/bold red]")
    console.print(message.format(**kwargs))


def show_plot_nonblocking(visualizer_method, *args, **kwargs) -> bool:
    """
    Displays a plot non-blockingly, allowing Ctrl+C to close and return to prompt.

    Args:
        visualizer_method: Visualizer method (e.g., Visualizer.plot_histogram).
        *args: Positional arguments for the method.
        **kwargs: Keyword arguments for the method.

    Returns:
        bool: True if closed with Enter, False if closed with Ctrl+C.
    """
    plt.ion()  # Enable interactive mode
    visualizer_method(*args, **kwargs)
    plt.draw()  # Draw the plot
    plt.pause(0.1)  # Brief pause to show plot
    try:
        input("Press Enter or Ctrl+C to continue...")
        plt.close()  # Close plot
        return True  # Closed with Enter
    except KeyboardInterrupt:
        plt.close()  # Close plot on Ctrl+C
        print_message("plot_closed_ctrl_c")
        return False  # Closed with Ctrl+C
    finally:
        plt.ioff()  # Disable interactive mode after closing


def format_params(args: Dict) -> str:
    """
    Formats experiment parameters for display, excluding visualization keys.
    """
    params = {
        k: v
        for k, v in args.items()
        if k
        not in [
            "visualization_type",
            "save_plot",
            "min_occurrences",
            "show_real",
            "show_imag",
            "hypergraph_config",
            "time_steps",
            "noise_stepped",
            "noise_start",
            "noise_end",
            "noise_steps",
            "z_prob_start",
            "z_prob_end",
            "i_prob_start",
            "i_prob_end",
            "t1_start",
            "t1_end",
            "t2_start",
            "t2_end",
        ]
    }
    return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)


def display_params_summary(args: Dict) -> None:
    """
    Displays a formatted summary of experiment parameters before running.
    """
    table = Table(
        title="Experiment Parameters", show_header=True, header_style="bold magenta"
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    params_to_display = {
        "Number of Qubits": args["num_qubits"],
        "State Type": args["state_type"],
        "Noise Type": args["noise_type"],
        "Noise Enabled": args["noise_enabled"],
        "Shots": args["shots"],
        "Simulation Mode": args["sim_mode"],
        "Error Rate": (
            args["error_rate"] if args["error_rate"] is not None else "Default"
        ),
        "Z Probability": args["z_prob"] if args["z_prob"] is not None else "Default",
        "I Probability": args["i_prob"] if args["i_prob"] is not None else "Default",
        "T1": args["t1"] if args["t1"] is not None else "Default",
        "T2": args["t2"] if args["t2"] is not None else "Default",
        "Custom Params": args["custom_params"] if args["custom_params"] else "None",
    }
    # Add stepped parameters if applicable
    if args.get("noise_stepped", False):
        params_to_display.update(
            {
                "Noise Stepped": "Yes",
                "Noise Start": args.get("noise_start", 0.0),
                "Noise End": args.get("noise_end", 0.5),
                "Noise Steps": args.get("noise_steps", 10),
            }
        )
        if "z_prob_start" in args:
            params_to_display["Z Prob Start"] = args["z_prob_start"]
            params_to_display["Z Prob End"] = args["z_prob_end"]
        if "i_prob_start" in args:
            params_to_display["I Prob Start"] = args["i_prob_start"]
            params_to_display["I Prob End"] = args["i_prob_end"]
        if "t1_start" in args:
            params_to_display["T1 Start"] = args["t1_start"]
            params_to_display["T1 End"] = args["t1_end"]
        if "t2_start" in args:
            params_to_display["T2 Start"] = args["t2_start"]
            params_to_display["T2 End"] = args["t2_end"]

    for param, value in params_to_display.items():
        table.add_row(param, str(value))
    console.print(table)


def switch_to_plot(args: Dict) -> Dict:
    """
    Switches the visualization type to 'plot' and collects related settings.
    """
    args["visualization_type"] = "plot"
    if args["sim_mode"] == "qasm":
        args["min_occurrences"] = input_handler.get_numeric_input(
            "min_occurrences_prompt", "0"
        )
    else:  # density mode
        real_imag = input_handler.get_input("real_imag_prompt", "a", ["r", "i", "a"])
        args["show_real"] = real_imag == "r"
        args["show_imag"] = real_imag == "i"
    return args


def collect_core_parameters(args: Dict) -> Dict:
    """
    Collects core experiment parameters from the user.
    """
    print_message("enter_parameters")

    # Collect number of qubits
    try:
        args["num_qubits"] = input_handler.get_numeric_input(
            "num_qubits_prompt", str(args["num_qubits"])
        )
    except ValueError:
        return collect_parameters(interactive=True)

    # Collect noise type with shortcuts and auto-correction
    noise_input = input_handler.get_input(
        "noise_type_prompt",
        default=args["noise_type"].lower(),
        valid_options=VALID_NOISE_TYPES + ["d", "p", "a", "z", "t", "b"],
        valid_options_display=VALID_NOISE_TYPES,
    )
    args["noise_type"] = noise_input.upper()
    args["noise_type"] = NOISE_SHORTCUTS.get(noise_input, args["noise_type"])

    # Collect state type
    args["state_type"] = input_handler.get_input(
        "state_type_prompt",
        default=args["state_type"].lower(),
        valid_options=VALID_STATE_TYPES,
    ).upper()

    # Collect noise enabled
    args["noise_enabled"] = input_handler.get_input(
        "noise_enabled_prompt",
        default=str(args["noise_enabled"]).lower(),
        valid_options=["y", "yes", "t", "true", "n", "no", "f", "false"],
    ) in ["y", "yes", "t", "true"]

    # Collect simulation mode
    args["sim_mode"] = input_handler.get_input(
        "sim_mode_prompt",
        default=args["sim_mode"].lower(),
        valid_options=["q", "d", "qasm", "density"],
    )
    args["sim_mode"] = (
        "qasm"
        if args["sim_mode"] in ["q", "qasm"]
        else (
            "density"
            if args["sim_mode"] in ["d", "density"]
            else args["sim_mode"].lower()
        )
    )

    # Collect shots
    try:
        args["shots"] = input_handler.get_numeric_input(
            "shots_prompt", str(args["shots"])
        )
    except ValueError:
        return collect_parameters(interactive=True)

    return args


def collect_stepped_noise_parameters(args: Dict) -> Dict:
    """
    Collects parameters for time-stepped noise simulations.
    """
    if not args["noise_enabled"]:
        return args

    # Ask if user wants to run a stepped simulation
    args["noise_stepped"] = input_handler.prompt_yes_no(
        "noise_stepped_prompt", default="n"
    )
    if not args["noise_stepped"]:
        return args

    # Collect number of steps
    args["noise_steps"] = input_handler.get_numeric_input(
        "noise_steps_prompt", "10", int
    )

    # Collect stepped parameters for error_rate
    if input_handler.prompt_yes_no("custom_error_rate_stepped_prompt", default="y"):
        args["noise_start"] = input_handler.get_numeric_input(
            "noise_start_prompt", "0.0", float
        )
        args["noise_end"] = input_handler.get_numeric_input(
            "noise_end_prompt", "0.5", float
        )

    # Collect stepped parameters for PHASE_FLIP noise
    if args["noise_type"] == "PHASE_FLIP":
        if input_handler.prompt_yes_no("custom_zi_probs_stepped_prompt", default="n"):
            args["z_prob_start"] = input_handler.get_numeric_input(
                "z_prob_start_prompt", "0.0", float
            )
            args["z_prob_end"] = input_handler.get_numeric_input(
                "z_prob_end_prompt", "0.5", float
            )
            args["i_prob_start"] = input_handler.get_numeric_input(
                "i_prob_start_prompt", "0.0", float
            )
            args["i_prob_end"] = input_handler.get_numeric_input(
                "i_prob_end_prompt", "0.5", float
            )

    # Collect stepped parameters for THERMAL_RELAXATION noise
    if args["noise_type"] == "THERMAL_RELAXATION":
        if input_handler.prompt_yes_no("custom_t1t2_stepped_prompt", default="n"):
            args["t1_start"] = (
                input_handler.get_numeric_input("t1_start_prompt", "100", float) * 1e-6
            )
            args["t1_end"] = (
                input_handler.get_numeric_input("t1_end_prompt", "50", float) * 1e-6
            )
            args["t2_start"] = (
                input_handler.get_numeric_input("t2_start_prompt", "80", float) * 1e-6
            )
            args["t2_end"] = (
                input_handler.get_numeric_input("t2_end_prompt", "40", float) * 1e-6
            )

    return args


def collect_visualization_settings(args: Dict) -> Dict:
    """
    Collects visualization-related settings from the user.
    """
    # Visualization selection
    viz_choice = input_handler.get_input(
        "viz_type_prompt", default="n", valid_options=["p", "h", "n"]
    )
    args["visualization_type"] = (
        "plot"
        if viz_choice in ["p", "plot"]
        else "hypergraph" if viz_choice in ["h", "hypergraph"] else "none"
    )
    if args["visualization_type"] != "none":
        args["save_plot"] = (
            input_handler.get_input("save_plot_prompt", default="").strip() or None
        )
        if args["visualization_type"] == "plot" and args["sim_mode"] == "qasm":
            try:
                args["min_occurrences"] = input_handler.get_numeric_input(
                    "min_occurrences_prompt", "0"
                )
            except ValueError:
                return collect_parameters(interactive=True)

    # Hypergraph-specific settings
    if args["visualization_type"] == "hypergraph":
        args["hypergraph_config"] = {}
        max_order = input_handler.get_numeric_input(
            "hypergraph_max_order_prompt", "2", int
        )
        args["hypergraph_config"]["max_order"] = max_order

        default_threshold = "0.1" if args["sim_mode"] == "qasm" else "0.01"
        threshold = input_handler.get_numeric_input(
            "hypergraph_threshold_prompt", default_threshold, float
        )
        args["hypergraph_config"]["threshold"] = threshold

        args["hypergraph_config"]["symmetry_analysis"] = input_handler.prompt_yes_no(
            "hypergraph_symmetry_analysis_prompt", default="n"
        )

        if "noise_stepped" in args and args["noise_stepped"]:
            args["hypergraph_config"]["plot_transitions"] = input_handler.prompt_yes_no(
                "hypergraph_plot_transitions_prompt", default="n"
            )
        else:
            args["hypergraph_config"]["plot_transitions"] = False

    return args


def collect_optional_parameters(args: Dict) -> Dict:
    """
    Collects optional parameters (error rate, Z/I probabilities, T1/T2, custom params) from the user.
    """
    # Optional parameters with confirmation
    if input_handler.prompt_yes_no("custom_error_rate_prompt", default="n"):
        args["error_rate"] = input_handler.get_numeric_input(
            "error_rate_value_prompt", str(DEFAULT_ERROR_RATE), float
        )
    if args["noise_type"] == "PHASE_FLIP" and input_handler.prompt_yes_no(
        "custom_zi_probs_prompt", default="n"
    ):
        args["z_prob"] = input_handler.get_numeric_input(
            "z_prob_value_prompt", "0.5", float
        )
        args["i_prob"] = input_handler.get_numeric_input(
            "i_prob_value_prompt", "0.5", float
        )
    if args["noise_type"] == "THERMAL_RELAXATION" and input_handler.prompt_yes_no(
        "custom_t1t2_prompt", default="n"
    ):
        args["t1"] = (
            input_handler.get_numeric_input("t1_value_prompt", "100", float) * 1e-6
        )
        args["t2"] = (
            input_handler.get_numeric_input("t2_value_prompt", "80", float) * 1e-6
        )
    if args["state_type"] == "CLUSTER" and input_handler.prompt_yes_no(
        "custom_lattice_prompt", default="n"
    ):
        if "custom_params" not in args:
            args["custom_params"] = {}
        lattice_type = input_handler.get_input(
            "lattice_type_prompt", "1d", ["1d", "2d"]
        )
        args["custom_params"]["lattice"] = lattice_type
    if input_handler.prompt_yes_no("custom_params_prompt", default="n"):
        custom_params_str = input_handler.get_input(
            "custom_params_value_prompt", default=""
        ).strip()
        try:
            args["custom_params"] = (
                json.loads(custom_params_str) if custom_params_str else None
            )
        except json.JSONDecodeError:
            print_message(
                "invalid_input", input="custom params", options=["valid JSON"]
            )
            return collect_parameters(interactive=True)

    return args


def validate_and_prompt(args: Dict) -> Dict:
    """
    Validates parameters and prompts the user for adjustments if needed.
    """
    # Single-qubit noise with multi-qubit system
    if args["noise_type"] in SINGLE_QUBIT_NOISE_TYPES and args["num_qubits"] > 1:
        choice = input_handler.get_input(
            "single_qubit_noise_prompt", "p", ["p", "switch", "c"]
        )
        if choice == "switch":
            print_message("suggested_multi_qubit_noise_types")
            new_noise = input_handler.get_input(
                "noise_type_prompt",
                "depolarizing",
                valid_options=[
                    "d",
                    "p",
                    "t",
                    "depolarizing",
                    "phase_flip",
                    "thermal_relaxation",
                ],
            )
            args["noise_type"] = (
                "DEPOLARIZING"
                if new_noise in ["d", "depolarizing"]
                else (
                    "PHASE_FLIP"
                    if new_noise in ["p", "phase_flip"]
                    else "THERMAL_RELAXATION"
                )
            )
            print_message("switched_noise_type", noise_type=args["noise_type"])
        elif choice == "c":
            print_message("config_cancelled")
            return collect_parameters(interactive=True)

    # Hypergraph + single-qubit noise
    if args["visualization_type"] == "hypergraph":
        if args["noise_type"] in SINGLE_QUBIT_NOISE_TYPES and args["num_qubits"] > 1:
            print_message(
                "hypergraph_single_qubit_warning",
                noise_type=args["noise_type"],
                num_qubits=args["num_qubits"],
            )
            choice = input_handler.get_input(
                "hypergraph_single_qubit_prompt", "p", ["p", "switch", "v"]
            )
            if choice == "switch":
                print_message("suggested_multi_qubit_noise_types")
                new_noise = input_handler.get_input(
                    "noise_type_prompt",
                    "depolarizing",
                    valid_options=[
                        "d",
                        "p",
                        "t",
                        "depolarizing",
                        "phase_flip",
                        "thermal_relaxation",
                    ],
                )
                args["noise_type"] = (
                    "DEPOLARIZING"
                    if new_noise in ["d", "depolarizing"]
                    else (
                        "PHASE_FLIP"
                        if new_noise in ["p", "phase_flip"]
                        else "THERMAL_RELAXATION"
                    )
                )
                print_message("switched_noise_type", noise_type=args["noise_type"])
            elif choice == "v":
                print_message("switched_to_plot")
                args = switch_to_plot(args)

    # Re-check if we switched to plot
    if args["visualization_type"] == "plot":
        args = switch_to_plot(args)

    # Single-qubit noise in density mode
    if (
        args["sim_mode"] == "density"
        and args["noise_type"] in SINGLE_QUBIT_NOISE_TYPES
        and args["noise_enabled"]
    ):
        choice = input_handler.get_input(
            "density_noise_prompt", "p", ["p", "switch", "c"]
        )
        if choice == "switch":
            print_message("suggested_multi_qubit_noise_types")
            new_noise = input_handler.get_input(
                "noise_type_prompt",
                "depolarizing",
                valid_options=[
                    "d",
                    "p",
                    "t",
                    "depolarizing",
                    "phase_flip",
                    "thermal_relaxation",
                ],
            )
            args["noise_type"] = (
                "DEPOLARIZING"
                if new_noise in ["d", "depolarizing"]
                else (
                    "PHASE_FLIP"
                    if new_noise in ["p", "phase_flip"]
                    else "THERMAL_RELAXATION"
                )
            )
            print_message("switched_noise_type", noise_type=args["noise_type"])
        elif choice == "p":
            args["noise_enabled"] = False
            print_message("noise_disabled")
        elif choice == "c":
            print_message("config_cancelled")
            return collect_parameters(interactive=True)

    # Hypergraph + density mode + no noise
    if (
        args["visualization_type"] == "hypergraph"
        and args["sim_mode"] == "density"
        and not args["noise_enabled"]
    ):
        print_message(
            "hypergraph_density_no_noise_warning", state_type=args["state_type"]
        )
        choice = input_handler.get_input(
            "hypergraph_density_no_noise_prompt", "p", ["p", "e", "v"]
        )
        if choice == "e":
            args["noise_enabled"] = True
            print_message("noise_enabled")
        elif choice == "v":
            print_message("switched_to_plot_density")
            args = switch_to_plot(args)

    # If user sets hypergraph transitions but not noise_stepped, disable transitions
    if (
        args["visualization_type"] == "hypergraph"
        and args.get("hypergraph_config", {}).get("plot_transitions", False)
        and not args.get("noise_stepped", False)
    ):
        print_message("hypergraph_transitions_dependency")
        args["hypergraph_config"]["plot_transitions"] = False

    return args


def collect_parameters(interactive: bool = True) -> Dict:
    """
    Collects experiment parameters either interactively or from command-line arguments.
    """
    args = apply_defaults({})

    if interactive:
        # Core
        args = collect_core_parameters(args)
        # Stepped noise
        args = collect_stepped_noise_parameters(args)
        # Visualization
        args = collect_visualization_settings(args)
        # Validate
        args = validate_and_prompt(args)
        # Optional
        args = collect_optional_parameters(args)

    return validate_parameters(args)


def run_and_visualize(
    args: Dict, experiment_id: str
) -> Tuple[QuantumCircuit, Union[Dict, DensityMatrix], bool]:
    """
    Runs the experiment and handles visualization.
    """
    # Filter out visualization-only keys
    args_for_experiment = {
        key: value
        for key, value in args.items()
        if key
        not in [
            "visualization_type",
            "save_plot",
            "min_occurrences",
            "show_real",
            "show_imag",
            "hypergraph_config",
            "noise_stepped",
            "noise_start",
            "noise_end",
            "noise_steps",
            "z_prob_start",
            "z_prob_end",
            "i_prob_start",
            "i_prob_end",
            "t1_start",
            "t1_end",
            "t2_start",
            "t2_end",
        ]
    }
    args_for_experiment["experiment_id"] = experiment_id

    # Handle time-stepped noise
    if args.get("noise_stepped", False) and args["noise_enabled"]:
        # Generate stepped parameters
        error_rates = (
            np.linspace(args["noise_start"], args["noise_end"], args["noise_steps"])
            if "noise_start" in args and "noise_end" in args
            else np.array([args["error_rate"]] * args["noise_steps"])
        )
        z_probs = (
            np.linspace(args["z_prob_start"], args["z_prob_end"], args["noise_steps"])
            if "z_prob_start" in args and "z_prob_end" in args
            else np.array(
                [args["z_prob"] if args["z_prob"] is not None else 0.0]
                * args["noise_steps"]
            )
        )
        i_probs = (
            np.linspace(args["i_prob_start"], args["i_prob_end"], args["noise_steps"])
            if "i_prob_start" in args and "i_prob_end" in args
            else np.array(
                [args["i_prob"] if args["i_prob"] is not None else 0.0]
                * args["noise_steps"]
            )
        )
        t1_values = (
            np.linspace(args["t1_start"], args["t1_end"], args["noise_steps"])
            if "t1_start" in args and "t1_end" in args
            else np.array(
                [args["t1"] if args["t1"] is not None else 100.0] * args["noise_steps"]
            )
        )
        t2_values = (
            np.linspace(args["t2_start"], args["t2_end"], args["noise_steps"])
            if "t2_start" in args and "t2_end" in args
            else np.array(
                [args["t2"] if args["t2"] is not None else 80.0] * args["noise_steps"]
            )
        )

        results = []
        for idx in tqdm(range(args["noise_steps"]), desc="Running stepped simulations"):
            print_message(
                "running_step",
                step=idx + 1,
                total=args["noise_steps"],
                error_rate=error_rates[idx],
                z_prob=z_probs[idx],
                i_prob=i_probs[idx],
                t1=t1_values[idx],
                t2=t2_values[idx],
            )
            args_for_experiment["error_rate"] = error_rates[idx]
            args_for_experiment["z_prob"] = (
                z_probs[idx] if z_probs[idx] != 0.0 else None
            )
            args_for_experiment["i_prob"] = (
                i_probs[idx] if i_probs[idx] != 0.0 else None
            )
            args_for_experiment["t1"] = (
                t1_values[idx] if t1_values[idx] != 100.0 else None
            )
            args_for_experiment["t2"] = (
                t2_values[idx] if t2_values[idx] != 80.0 else None
            )

            qc, result = run_experiment(**args_for_experiment)
            results.append(result)
        time_steps = error_rates.tolist()  # Use error rates as "time steps"
    else:
        # Single run
        qc, result = run_experiment(**args_for_experiment)
        results = result
        time_steps = None

    # Save results (timestamp first)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"{timestamp}_experiment_results_{args['num_qubits']}q_{args['state_type']}_"
        f"{args['noise_type']}_{args['sim_mode']}.json"
    )

    ExperimentUtils.save_results(
        results[-1] if args.get("noise_stepped", False) else results,
        circuit=qc,
        experiment_params=args_for_experiment,
        filename=filename,
        experiment_id=experiment_id,
    )
    print_message("experiment_completed", filename=filename)

    # Debug log to confirm visualization type
    print_message("debug_viz_type", viz_type=args["visualization_type"])

    # If noise stepped is enabled and visualization type is "plot", use the last result;
    # otherwise, pass the entire results (e.g., for hypergraph visualization)
    if args.get("noise_stepped", False) and args["visualization_type"] == "plot":
        viz_result = results[-1]
    else:
        viz_result = results

    # Handle visualization
    hypergraph_config = args.get("hypergraph_config", {})
    plot_closed_with_ctrl_c = handle_visualization(
        viz_result,
        args,
        args["sim_mode"],
        args["state_type"],
        args["noise_type"],
        args["noise_enabled"],
        args.get("save_plot"),
        show_plot_nonblocking,
        config=hypergraph_config,
        time_steps=time_steps,
    )

    return qc, results, plot_closed_with_ctrl_c


@click.command()
@click.option("--num-qubits", type=int, help="Number of qubits for the experiment")
@click.option(
    "--state-type",
    type=click.Choice(VALID_STATE_TYPES, case_sensitive=False),
    help="Type of quantum state",
)
@click.option(
    "--noise-type",
    type=click.Choice(VALID_NOISE_TYPES, case_sensitive=False),
    help="Type of noise model",
)
@click.option(
    "--noise-enabled/--no-noise", default=True, help="Enable or disable noise"
)
@click.option("--shots", type=int, help="Number of shots for qasm simulation")
@click.option(
    "--sim-mode",
    type=click.Choice(["qasm", "density"], case_sensitive=False),
    help="Simulation mode",
)
@click.option("--error-rate", type=float, help="Custom error rate for noise models")
@click.option("--z-prob", type=float, help="Z probability for PHASE_FLIP noise")
@click.option("--i-prob", type=float, help="I probability for PHASE_FLIP noise")
@click.option(
    "--t1", type=float, help="T1 relaxation time (µs) for THERMAL_RELAXATION noise"
)
@click.option(
    "--t2", type=float, help="T2 dephasing time (µs) for THERMAL_RELAXATION noise"
)
@click.option(
    "--interactive/--no-interactive", default=True, help="Run in interactive mode"
)
def main(
    num_qubits: Optional[int],
    state_type: Optional[str],
    noise_type: Optional[str],
    noise_enabled: bool,
    shots: Optional[int],
    sim_mode: Optional[str],
    error_rate: Optional[float],
    z_prob: Optional[float],
    i_prob: Optional[float],
    t1: Optional[float],
    t2: Optional[float],
    interactive: bool,
):
    """
    Quantum Experiment Interactive Runner

    A CLI tool to run quantum experiments with configurable parameters,
    supporting interactive and non-interactive modes.
    """
    if interactive:
        interactive_experiment()
    else:
        # Non-interactive mode
        args = {
            "num_qubits": num_qubits,
            "state_type": state_type,
            "noise_type": noise_type,
            "noise_enabled": noise_enabled,
            "shots": shots,
            "sim_mode": sim_mode,
            "visualization_type": "none",
            "save_plot": None,
            "min_occurrences": 0,
            "show_real": False,
            "show_imag": False,
            "error_rate": error_rate,
            "z_prob": z_prob,
            "i_prob": i_prob,
            "t1": t1,
            "t2": t2,
            "custom_params": None,
        }
        args = apply_defaults(args)
        experiment_id = str(uuid.uuid4())
        qc, result, _ = run_and_visualize(args, experiment_id)


def interactive_experiment():
    """
    Runs the quantum experiment interactively with rerun and skip options.
    """
    while True:
        print_message("welcome")
        print_message("choose_option")
        print_message("skip_option")
        print_message("new_option")
        print_message("quit_option")

        choice = input_handler.get_input("your_choice", "s", ["s", "n", "q"])

        if choice == "s":
            print_message("running_with_defaults")
            viz_choice = input_handler.get_input(
                "viz_type_prompt", "p", ["p", "h", "n"]
            )
            args = collect_parameters(interactive=True)
            args["visualization_type"] = (
                "plot"
                if viz_choice in ["p", "plot"]
                else "hypergraph" if viz_choice in ["h", "hypergraph"] else "none"
            )
            if args["visualization_type"] != "none":
                args["save_plot"] = (
                    input_handler.get_input("save_plot_prompt", "").strip() or None
                )
                if args["visualization_type"] == "plot" and args["sim_mode"] == "qasm":
                    args["min_occurrences"] = input_handler.get_numeric_input(
                        "min_occurrences_prompt", "0"
                    )
        elif choice == "n":
            args = collect_parameters(interactive=True)
        elif choice == "q":
            print_message("goodbye")
            return
        else:
            print_message("invalid_choice")
            continue

        # Display parameter summary
        display_params_summary(args)

        # Confirm before running
        if input_handler.get_input("proceed_prompt", "y", ["y", "n"]) != "y":
            print_message("params_discarded")
            continue

        experiment_id = str(uuid.uuid4())
        qc, result, plot_closed_with_ctrl_c = run_and_visualize(args, experiment_id)

        # Rerun prompt
        while True:
            print_message("current_params", params=format_params(args))
            if plot_closed_with_ctrl_c:
                print_message("rerun_plot_prompt")
                rerun_choice = input_handler.get_input(
                    "rerun_choice_prompt", "y", ["y", "n"]
                )
                if rerun_choice == "y":
                    experiment_id = str(uuid.uuid4())
                    qc, result, plot_closed_with_ctrl_c = run_and_visualize(
                        args, experiment_id
                    )
                    continue
                else:
                    plot_closed_with_ctrl_c = False  # Reset flag

            next_choice = input_handler.get_input("rerun_prompt", "r", ["r", "n", "q"])
            if next_choice == "r":
                print_message("rerun_same")
                logger.log_with_experiment_id(
                    logger_instance,
                    "info",
                    f"Rerunning experiment with {args['num_qubits']} qubits, {args['state_type']} state, "
                    f"{'with' if args['noise_enabled'] else 'without'} {args['noise_type']} noise",
                    experiment_id,
                    extra_info={
                        "num_qubits": args["num_qubits"],
                        "state_type": args["state_type"],
                        "noise_type": args["noise_type"],
                        "noise_enabled": args["noise_enabled"],
                        "sim_mode": args["sim_mode"],
                    },
                )
                experiment_id = str(uuid.uuid4())
                qc, result, plot_closed_with_ctrl_c = run_and_visualize(
                    args, experiment_id
                )
            elif next_choice == "n":
                print_message("restart_params")
                break
            else:  # 'q'
                print_message("goodbye")
                return


if __name__ == "__main__":
    main()
