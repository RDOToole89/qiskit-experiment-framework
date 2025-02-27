#!/usr/bin/env python3
# src/main.py

"""
Interactive script to run quantum experiments, designed for extensibility and research integration.

This script provides an interactive CLI to execute quantum experiments, supporting:
- Dynamic, case-insensitive parameter selection for states, noise models, and simulation modes.
- Logging and structured results saving for hypergraph or decoherence analysis.
- Optional, non-blocking visualization via histograms, density matrices, or hypergraphs,
  closable with Ctrl+C, with save and filtering options.
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from src.run_experiment import run_experiment
from src.utils import logger, results as ExperimentUtils
from src.visualization import Visualizer
from src.config.params import apply_defaults, validate_parameters
from src.config.constants import VALID_NOISE_TYPES, VALID_STATE_TYPES
from src.config.defaults import DEFAULT_ERROR_RATE
from src.noise_models.noise_factory import NOISE_CLASSES  # Import NOISE_CLASSES to identify single-qubit noise

# Suppress Qiskit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logger with structured output
logger_instance = logger.setup_logger(
    log_level="INFO",
    log_to_file=True,
    log_to_console=True,
    structured_log_file="logs/structured_logs.json"
)

# Initialize Rich console for better formatting
console = Console()

def get_input(prompt: str, default: str, valid_options: Optional[list] = None) -> str:
    """
    Helper to get user input with case-insensitive handling and validation.

    Args:
        prompt (str): Input prompt message.
        default (str): Default value if user presses Enter.
        valid_options (list, optional): List of valid options for validation.

    Returns:
        str: User input or default, normalized to lowercase.
    """
    while True:
        try:
            user_input = console.input(f"[bold cyan]{prompt}[/bold cyan] ").strip().lower() or default.lower()
            if not valid_options or user_input in [opt.lower() for opt in valid_options]:
                return user_input
            console.print(f"[bold red]⚠️ Invalid input: '{user_input}'. Please choose from {valid_options}.[/bold red]")
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Operation cancelled, returning to prompt...[/bold yellow]")
            return default.lower()

def format_params(args: Dict) -> str:
    """
    Formats experiment parameters for display, excluding visualization keys.

    Args:
        args (Dict): Experiment parameters.

    Returns:
        str: Formatted string of parameters.
    """
    params = {
        k: v
        for k, v in args.items()
        if k not in ["visualization_type", "save_plot", "min_occurrences", "show_real", "show_imag"]
    }
    return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)

def display_params_summary(args: Dict) -> None:
    """
    Displays a formatted summary of experiment parameters before running.

    Args:
        args (Dict): Experiment parameters.
    """
    table = Table(title="Experiment Parameters", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    params_to_display = {
        "Number of Qubits": args["num_qubits"],
        "State Type": args["state_type"],
        "Noise Type": args["noise_type"],
        "Noise Enabled": args["noise_enabled"],
        "Shots": args["shots"],
        "Simulation Mode": args["sim_mode"],
        "Error Rate": args["error_rate"] if args["error_rate"] is not None else "Default",
        "Z Probability": args["z_prob"] if args["z_prob"] is not None else "Default",
        "I Probability": args["i_prob"] if args["i_prob"] is not None else "Default",
        "T1": args["t1"] if args["t1"] is not None else "Default",
        "T2": args["t2"] if args["t2"] is not None else "Default",
        "Custom Params": args["custom_params"] if args["custom_params"] else "None"
    }
    for param, value in params_to_display.items():
        table.add_row(param, str(value))
    console.print(table)

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
        input("Press Enter or Ctrl+C to continue...")  # Wait for user input or interrupt
        plt.close()  # Close plot
        return True  # Closed with Enter
    except KeyboardInterrupt:
        plt.close()  # Close plot on Ctrl+C
        console.print("\n[bold yellow]Plot closed with Ctrl+C, returning to prompt...[/bold yellow]")
        return False  # Closed with Ctrl+C
    finally:
        plt.ioff()  # Disable interactive mode after closing

def collect_parameters(interactive: bool = True) -> Dict:
    """
    Collects experiment parameters either interactively or from command-line arguments.

    Args:
        interactive (bool): Whether to collect parameters interactively.

    Returns:
        Dict: Collected experiment parameters.
    """
    args = apply_defaults({})

    if interactive:
        console.print("\n[bold blue]🔹 Enter your experiment parameters below:[/bold blue]\n")

        # Collect number of qubits first
        args["num_qubits"] = int(
            get_input(f"Number of qubits [{args['num_qubits']}]: ", str(args["num_qubits"]))
        )

        # Collect noise type with shortcuts and auto-correction
        noise_input = get_input(
            f"Enter noise type {VALID_NOISE_TYPES} (d/p/a/z/t/b) [{args['noise_type'].lower()}]: ",
            args["noise_type"].lower(),
        )
        # Convert noise type to uppercase immediately for consistency
        args["noise_type"] = noise_input.upper()
        # Map shortcuts to full names (e.g., "b" -> "BIT_FLIP")
        noise_shortcuts = {
            "d": "DEPOLARIZING",
            "p": "PHASE_FLIP",
            "a": "AMPLITUDE_DAMPING",
            "z": "PHASE_DAMPING",
            "t": "THERMAL_RELAXATION",
            "b": "BIT_FLIP",
        }
        args["noise_type"] = noise_shortcuts.get(noise_input, args["noise_type"])

        # Check if noise type is single-qubit and num_qubits > 1
        single_qubit_noise_types = ["AMPLITUDE_DAMPING", "PHASE_DAMPING", "BIT_FLIP"]
        if args["noise_type"] in single_qubit_noise_types and args["num_qubits"] > 1:
            console.print(
                f"[bold yellow]⚠️ Warning: {args['noise_type']} noise is designed for single-qubit systems, "
                f"but you requested {args['num_qubits']} qubits. This noise will only be applied to "
                "single-qubit gates ('id', 'u1', 'u2', 'u3').[/bold yellow]"
            )
            choice = get_input(
                "Would you like to proceed with this configuration, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or cancel? (p/switch/c) [p]: ",
                "p",
                ["p", "switch", "c"]
            )
            if choice == "switch":
                console.print(
                    "[bold blue]Suggested multi-qubit noise types: DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION[/bold blue]"
                )
                new_noise = get_input(
                    "Enter a new noise type (d/p/t for DEPOLARIZING/PHASE_FLIP/THERMAL_RELAXATION) [depolarizing]: ",
                    "depolarizing",
                    ["d", "p", "t", "depolarizing", "phase_flip", "thermal_relaxation"]
                )
                args["noise_type"] = (
                    "DEPOLARIZING" if new_noise in ["d", "depolarizing"]
                    else "PHASE_FLIP" if new_noise in ["p", "phase_flip"]
                    else "THERMAL_RELAXATION"
                )
                console.print(f"[bold green]Switched noise type to {args['noise_type']}.[/bold green]")
            elif choice == "c":
                console.print("[bold yellow]Configuration cancelled. Returning to prompt...[/bold yellow]")
                return collect_parameters(interactive=True)  # Restart parameter collection

        # Visualization selection
        viz_choice = get_input(
            "\n🎨 Choose visualization type (p/plot, h/hypergraph, n/none) [n]: ",
            "n",
            ["p", "h", "n"],
        )
        args["visualization_type"] = (
            "plot" if viz_choice in ["p", "plot"]
            else "hypergraph" if viz_choice in ["h", "hypergraph"]
            else "none"
        )
        if args["visualization_type"] != "none":
            args["save_plot"] = (
                get_input("Enter path to save plot (press Enter for display): ", "").strip()
                or None
            )

        # Check for hypergraph visualization compatibility: Single-qubit noise with multi-qubit states
        if args["visualization_type"] == "hypergraph":
            if args["noise_type"] in single_qubit_noise_types and args["num_qubits"] > 1:
                console.print(
                    f"[bold yellow]⚠️ Warning: {args['noise_type']} noise with {args['num_qubits']} qubits may not be meaningful for hypergraph visualization. "
                    "Single-qubit noise only applies to single-qubit gates and won't affect multi-qubit correlations (e.g., entanglement between qubits). "
                    f"The hypergraph may only show the ideal correlations of the state without noise impact.[/bold yellow]"
                )
                choice = get_input(
                    "Would you like to proceed with this configuration, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or change visualization type? (p/switch/v) [p]: ",
                    "p",
                    ["p", "switch", "v"]
                )
                if choice == "switch":
                    console.print(
                        "[bold blue]Suggested multi-qubit noise types: DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION[/bold blue]"
                    )
                    new_noise = get_input(
                        "Enter a new noise type (d/p/t for DEPOLARIZING/PHASE_FLIP/THERMAL_RELAXATION) [depolarizing]: ",
                        "depolarizing",
                        ["d", "p", "t", "depolarizing", "phase_flip", "thermal_relaxation"]
                    )
                    args["noise_type"] = (
                        "DEPOLARIZING" if new_noise in ["d", "depolarizing"]
                        else "PHASE_FLIP" if new_noise in ["p", "phase_flip"]
                        else "THERMAL_RELAXATION"
                    )
                    console.print(f"[bold green]Switched noise type to {args['noise_type']}.[/bold green]")
                elif choice == "v":
                    console.print("[bold blue]Switching visualization type to 'plot' (histogram/density matrix).[/bold blue]")
                    args["visualization_type"] = "plot"

        # Set visualization-specific settings after any potential switch
        if args["visualization_type"] == "plot":
            if args["sim_mode"] == "qasm":
                args["min_occurrences"] = int(
                    get_input(f"Minimum occurrences [0]: ", "0") or 0
                )
            else:  # density mode
                real_imag = get_input(
                    "Show real (r), imaginary (i), or absolute (a) values? [a]: ",
                    "a",
                    ["r", "i", "a"],
                )
                args["show_real"] = real_imag == "r"
                args["show_imag"] = real_imag == "i"

        # Core parameters
        args["state_type"] = get_input(
            f"State type {VALID_STATE_TYPES} [{args['state_type'].lower()}]: ",
            args["state_type"].lower(),
            VALID_STATE_TYPES,
        ).upper()
        args["noise_enabled"] = get_input(
            f"Enable noise? (y/yes/t/true, n/no/f/false) [{str(args['noise_enabled']).lower()}]: ",
            str(args["noise_enabled"]).lower(),
            ["y", "yes", "t", "true", "n", "no", "f", "false"],
        ) in ["y", "yes", "t", "true"]

        # Collect simulation mode with shortcuts q/d for qasm/density
        sim_input = get_input(
            f"Simulation mode (q/qasm, d/density) [{args['sim_mode'].lower()}]: ",
            args["sim_mode"].lower(),
            ["q", "d", "qasm", "density"],
        )
        args["sim_mode"] = (
            "qasm" if sim_input in ["q", "qasm"]
            else "density" if sim_input in ["d", "density"]
            else args["sim_mode"].lower()
        )

        # Check if noise type is single-qubit and sim_mode is density with noise enabled
        if (
            args["sim_mode"] == "density"
            and args["noise_type"] in single_qubit_noise_types
            and args["noise_enabled"]
        ):
            console.print(
                f"[bold yellow]⚠️ Warning: {args['noise_type']} noise only applies to single-qubit gates, which are skipped in density matrix simulation mode. "
                "No noise will be applied with this configuration.[/bold yellow]"
            )
            choice = get_input(
                "Would you like to proceed with noise disabled, switch to a multi-qubit noise type (e.g., DEPOLARIZING), or cancel? (p/switch/c) [p]: ",
                "p",
                ["p", "switch", "c"]
            )
            if choice == "switch":
                console.print(
                    "[bold blue]Suggested multi-qubit noise types: DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION[/bold blue]"
                )
                new_noise = get_input(
                    "Enter a new noise type (d/p/t for DEPOLARIZING/PHASE_FLIP/THERMAL_RELAXATION) [depolarizing]: ",
                    "depolarizing",
                    ["d", "p", "t", "depolarizing", "phase_flip", "thermal_relaxation"]
                )
                args["noise_type"] = (
                    "DEPOLARIZING" if new_noise in ["d", "depolarizing"]
                    else "PHASE_FLIP" if new_noise in ["p", "phase_flip"]
                    else "THERMAL_RELAXATION"
                )
                console.print(f"[bold green]Switched noise type to {args['noise_type']}.[/bold green]")
            elif choice == "p":
                args["noise_enabled"] = False  # Disable noise if proceeding
                console.print("[bold yellow]Noise has been disabled for this configuration.[/bold yellow]")
            elif choice == "c":
                console.print("[bold yellow]Configuration cancelled. Returning to prompt...[/bold yellow]")
                return collect_parameters(interactive=True)  # Restart parameter collection

        # Check for hypergraph visualization compatibility: Density mode with no noise
        if args["visualization_type"] == "hypergraph" and args["sim_mode"] == "density" and not args["noise_enabled"]:
            console.print(
                f"[bold yellow]⚠️ Warning: Hypergraph visualization in density matrix simulation mode with no noise enabled may not be insightful. "
                f"The hypergraph will only show the ideal correlations of the {args['state_type']} state without noise effects.[/bold yellow]"
            )
            choice = get_input(
                "Would you like to proceed with this configuration, enable noise, or change visualization type? (p/e/v) [p]: ",
                "p",
                ["p", "e", "v"]
            )
            if choice == "e":
                args["noise_enabled"] = True
                console.print("[bold green]Noise has been enabled for this configuration.[/bold green]")
            elif choice == "v":
                console.print("[bold blue]Switching visualization type to 'plot' (density matrix).[/bold blue]")
                args["visualization_type"] = "plot"
                real_imag = get_input(
                    "Show real (r), imaginary (i), or absolute (a) values? [a]: ",
                    "a",
                    ["r", "i", "a"],
                )
                args["show_real"] = real_imag == "r"
                args["show_imag"] = real_imag == "i"

        args["shots"] = int(
            get_input(f"Number of shots [{args['shots']}]: ", str(args["shots"]))
        )

        # Optional parameters with confirmation
        if get_input("Set custom error rate? (y/n) [n]: ", "n", ["y", "n"]) == "y":
            args["error_rate"] = float(
                get_input(f"Error rate [{DEFAULT_ERROR_RATE}]: ", str(DEFAULT_ERROR_RATE))
            )
        if (
            args["noise_type"] == "PHASE_FLIP"
            and get_input("Set custom Z/I probabilities? (y/n) [n]: ", "n", ["y", "n"]) == "y"
        ):
            args["z_prob"] = float(get_input("Z probability for PHASE_FLIP [0.5]: ", "0.5"))
            args["i_prob"] = float(get_input("I probability for PHASE_FLIP [0.5]: ", "0.5"))
        if (
            args["noise_type"] == "THERMAL_RELAXATION"
            and get_input("Set custom T1/T2? (y/n) [n]: ", "n", ["y", "n"]) == "y"
        ):
            args["t1"] = (
                float(get_input("T1 for THERMAL_RELAXATION (µs) [100]: ", "100")) * 1e-6
            )
            args["t2"] = (
                float(get_input("T2 for THERMAL_RELAXATION (µs) [80]: ", "80")) * 1e-6
            )
        if (
            args["state_type"] == "CLUSTER"
            and get_input("Set custom lattice? (y/n) [n]: ", "n", ["y", "n"]) == "y"
        ):
            if "custom_params" not in args:
                args["custom_params"] = {}
            lattice_type = get_input("Lattice type (1d/2d) [1d]: ", "1d", ["1d", "2d"])
            args["custom_params"]["lattice"] = lattice_type
        if get_input("Set custom params? (y/n) [n]: ", "n", ["y", "n"]) == "y":
            custom_params_str = get_input(
                "Enter custom params as JSON (press Enter for none): ", ""
            ).strip()
            args["custom_params"] = (
                json.loads(custom_params_str) if custom_params_str else None
            )

    return validate_parameters(args)

def run_and_visualize(args: Dict, experiment_id: str) -> Tuple[QuantumCircuit, Union[Dict, DensityMatrix], bool]:
    """
    Runs the experiment and handles visualization.

    Args:
        args (Dict): Experiment parameters.
        experiment_id (str): Unique identifier for the experiment.

    Returns:
        Tuple[QuantumCircuit, Union[Dict, DensityMatrix], bool]: Circuit, result, and flag indicating if plot was closed with Ctrl+C.
    """
    args_for_experiment = {
        key: value
        for key, value in args.items()
        if key not in ["visualization_type", "save_plot", "min_occurrences", "show_real", "show_imag"]
    }
    args_for_experiment["experiment_id"] = experiment_id

    # Run the experiment with a progress spinner
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Running experiment..."),
        transient=True
    ) as progress:
        task = progress.add_task("Experiment", total=None)
        qc, result = run_experiment(**args_for_experiment)
        progress.update(task, completed=True)

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"results/experiment_results_{args['num_qubits']}q_{args['state_type']}_"
        f"{args['noise_type']}_{args['sim_mode']}_{timestamp}.json"
    )
    ExperimentUtils.save_results(result, circuit=qc, experiment_params=args_for_experiment, filename=filename, experiment_id=experiment_id)
    console.print(f"\n[bold green]✅ Experiment completed successfully![/bold green]\n📁 Results saved in `{filename}`")

    # Debug log to confirm visualization type
    console.print(f"[bold blue]Debug: Visualization type is {args['visualization_type']}[/bold blue]")

    # Handle visualization non-blockingly
    plot_closed_with_ctrl_c = False
    if args["visualization_type"] == "plot":
        if args["sim_mode"] == "qasm":
            if args["save_plot"]:
                Visualizer.plot_histogram(
                    result["counts"],
                    state_type=args["state_type"],
                    noise_type=args["noise_type"] if args["noise_enabled"] else None,
                    noise_enabled=args["noise_enabled"],
                    save_path=args["save_plot"],
                    min_occurrences=args["min_occurrences"],
                    num_qubits=args["num_qubits"],
                )
            else:
                plot_closed_with_ctrl_c = not show_plot_nonblocking(
                    Visualizer.plot_histogram,
                    result["counts"],
                    state_type=args["state_type"],
                    noise_type=args["noise_type"] if args["noise_enabled"] else None,
                    noise_enabled=args["noise_enabled"],
                    min_occurrences=args["min_occurrences"],
                    num_qubits=args["num_qubits"],
                )
        else:
            args["show_real"] = args.get("show_real", False)
            args["show_imag"] = args.get("show_imag", False)
            if args["save_plot"]:
                Visualizer.plot_density_matrix(
                    result,
                    cmap="viridis",
                    show_real=args["show_real"],
                    show_imag=args["show_imag"],
                    save_path=args["save_plot"],
                    state_type=args["state_type"],
                    noise_type=args["noise_type"] if args["noise_enabled"] else None,
                )
            else:
                plot_closed_with_ctrl_c = not show_plot_nonblocking(
                    Visualizer.plot_density_matrix,
                    result,
                    cmap="viridis",
                    show_real=args["show_real"],
                    show_imag=args["show_imag"],
                    state_type=args["state_type"],
                    noise_type=args["noise_type"] if args["noise_enabled"] else None,
                )
    elif args["visualization_type"] == "hypergraph":
        correlation_data = (
            result["counts"]
            if args["sim_mode"] == "qasm"
            else (
                {"density": np.abs(result.data).tolist()}
                if isinstance(result, DensityMatrix)
                else (
                    result.get("hypergraph", {}).get("correlations", {})
                    if isinstance(result, dict)
                    else {}
                )
            )
        )
        if args["save_plot"]:
            Visualizer.plot_hypergraph(
                correlation_data,
                state_type=args["state_type"],
                noise_type=args["noise_type"] if args["noise_enabled"] else None,
                save_path=args["save_plot"],
            )
        else:
            plot_closed_with_ctrl_c = not show_plot_nonblocking(
                Visualizer.plot_hypergraph,
                correlation_data,
                state_type=args["state_type"],
                noise_type=args["noise_type"] if args["noise_enabled"] else None,
            )

    return qc, result, plot_closed_with_ctrl_c

@click.command()
@click.option("--num-qubits", type=int, help="Number of qubits for the experiment")
@click.option("--state-type", type=click.Choice(VALID_STATE_TYPES, case_sensitive=False), help="Type of quantum state")
@click.option("--noise-type", type=click.Choice(VALID_NOISE_TYPES, case_sensitive=False), help="Type of noise model")
@click.option("--noise-enabled/--no-noise", default=True, help="Enable or disable noise")
@click.option("--shots", type=int, help="Number of shots for qasm simulation")
@click.option("--sim-mode", type=click.Choice(["qasm", "density"], case_sensitive=False), help="Simulation mode")
@click.option("--error-rate", type=float, help="Custom error rate for noise models")
@click.option("--z-prob", type=float, help="Z probability for PHASE_FLIP noise")
@click.option("--i-prob", type=float, help="I probability for PHASE_FLIP noise")
@click.option("--t1", type=float, help="T1 relaxation time (µs) for THERMAL_RELAXATION noise")
@click.option("--t2", type=float, help="T2 dephasing time (µs) for THERMAL_RELAXATION noise")
@click.option("--interactive/--no-interactive", default=True, help="Run in interactive mode")
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
    interactive: bool
):
    """
    Quantum Experiment Interactive Runner

    A CLI tool to run quantum experiments with configurable parameters, supporting interactive and non-interactive modes.
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

    Users can choose default settings, manually enter parameters, or modify them after an experiment.
    Results are saved and optionally visualized with non-blocking plots, closable with Ctrl+C.
    """
    while True:
        console.print("\n[bold green]🚀 Welcome to the Quantum Experiment Interactive Runner![/bold green]")
        console.print("🔹 Choose an option:")
        console.print("🔄 Press 's' to skip and use default settings")
        console.print("🆕 Press 'n' to enter parameters manually")
        console.print("❌ Press 'q' to quit")

        choice = get_input("➡️ Your choice: ", "s", ["s", "n", "q"])

        if choice == "s":
            console.print("\n[bold blue]⚡ Running with default configuration...[/bold blue]\n")
            # Prompt for quick visualization choice
            viz_choice = get_input(
                "Show visualization? (p/plot, h/hypergraph, n/none) [p]: ",
                "p",
                ["p", "h", "n"],
            )
            args = collect_parameters(interactive=True)
            args["visualization_type"] = (
                "plot" if viz_choice in ["p", "plot"]
                else "hypergraph" if viz_choice in ["h", "hypergraph"]
                else "none"
            )
            if args["visualization_type"] != "none":
                args["save_plot"] = (
                    get_input("Enter path to save (press Enter for display): ", "").strip()
                    or None
                )
                if args["visualization_type"] == "plot" and args["sim_mode"] == "qasm":
                    args["min_occurrences"] = int(
                        get_input(f"Minimum occurrences [0]: ", "0") or 0
                    )
        elif choice == "n":
            args = collect_parameters(interactive=True)
        elif choice == "q":
            console.print("\n[bold yellow]👋 Exiting Quantum Experiment Runner. Goodbye![/bold yellow]")
            return
        else:
            console.print("[bold red]⚠️ Invalid choice! Please enter s, n, or q.[/bold red]")
            continue

        # Display parameter summary
        display_params_summary(args)

        # Confirm before running
        if get_input("Proceed with these parameters? (y/n) [y]: ", "y", ["y", "n"]) != "y":
            console.print("[bold yellow]Parameters discarded. Returning to prompt...[/bold yellow]")
            continue

        experiment_id = str(uuid.uuid4())
        qc, result, plot_closed_with_ctrl_c = run_and_visualize(args, experiment_id)

        # Rerun prompt
        while True:
            params_str = format_params(args)
            console.print(f"\n[bold blue]🔄 Current parameters:[/bold blue] {params_str}")
            if plot_closed_with_ctrl_c:
                console.print("[bold yellow]Plot was closed with Ctrl+C. Would you like to run the experiment again with the same parameters?[/bold yellow]")
                rerun_choice = get_input("Run again? (y/n) [y]: ", "y", ["y", "n"])
                if rerun_choice == "y":
                    experiment_id = str(uuid.uuid4())
                    qc, result, plot_closed_with_ctrl_c = run_and_visualize(args, experiment_id)
                    continue
                else:
                    plot_closed_with_ctrl_c = False  # Reset flag

            next_choice = get_input(
                "\n➡️ Rerun? (r/same, n/new, q/quit): ", "r", ["r", "n", "q"]
            )
            if next_choice == "r":
                console.print("\n[bold blue]🔁 Rerunning with same parameters...[/bold blue]\n")
                logger.log_with_experiment_id(
                    logger_instance, "info",
                    f"Rerunning experiment with {args['num_qubits']} qubits, {args['state_type']} state, "
                    f"{'with' if args['noise_enabled'] else 'without'} {args['noise_type']} noise",
                    experiment_id,
                    extra_info={
                        "num_qubits": args["num_qubits"],
                        "state_type": args['state_type'],
                        "noise_type": args["noise_type"],
                        "noise_enabled": args["noise_enabled"],
                        "sim_mode": args["sim_mode"]
                    }
                )
                experiment_id = str(uuid.uuid4())
                qc, result, plot_closed_with_ctrl_c = run_and_visualize(args, experiment_id)
            elif next_choice == "n":
                console.print("\n[bold blue]🆕 Restarting parameter selection...[/bold blue]\n")
                break
            else:  # 'q'
                console.print("\n[bold yellow]👋 Exiting Quantum Experiment Runner. Goodbye![/bold yellow]")
                return

if __name__ == "__main__":
    main()
