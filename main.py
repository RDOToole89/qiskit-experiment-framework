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
"""

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
from typing import Optional, Dict
from qiskit.quantum_info import DensityMatrix
from src.run_experiment import run_experiment
from src.utils import ExperimentUtils

from src.visualization import Visualizer
from src.config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
    DEFAULT_ERROR_RATE,
)

# Suppress Qiskit deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# One-letter shortcuts for noise types (case-insensitive)
NOISE_SHORTCUTS = {
    "d": "DEPOLARIZING",
    "p": "PHASE_FLIP",
    "a": "AMPLITUDE_DAMPING",
    "z": "PHASE_DAMPING",
    "t": "THERMAL_RELAXATION",
    "b": "BIT_FLIP",
}
VALID_NOISE_TYPES = list(NOISE_SHORTCUTS.values())
VALID_STATE_TYPES = ["GHZ", "W", "CLUSTER"]  # Matches updated state_preparation.py

# Configure logger
logger = ExperimentUtils.setup_logger()


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
            user_input = input(f"{prompt} ").strip().lower() or default.lower()
            if not valid_options or user_input in [
                opt.lower() for opt in valid_options
            ]:
                return user_input
            print(
                f"⚠️ Invalid input: '{user_input}'. Please choose from {valid_options}."
            )
        except KeyboardInterrupt:
            print("\nOperation cancelled, returning to prompt...")
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
        if k
        not in [
            "visualization_type",
            "save_plot",
            "min_occurrences",
            "show_real",
            "show_imag",
        ]
    }
    return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None)


def show_plot_nonblocking(visualizer_method, *args, **kwargs):
    """
    Displays a plot non-blockingly, allowing Ctrl+C to close and return to prompt.

    Args:
        visualizer_method: Visualizer method (e.g., Visualizer.plot_histogram).
        *args: Positional arguments for the method.
        **kwargs: Keyword arguments for the method.
    """
    plt.ion()  # Enable interactive mode
    visualizer_method(*args, **kwargs)
    plt.draw()  # Draw the plot
    plt.pause(0.1)  # Brief pause to show plot
    try:
        input(
            "Press Enter or Ctrl+C to continue..."
        )  # Wait for user input or interrupt
    except KeyboardInterrupt:
        plt.close()  # Close plot on Ctrl+C
        print("\nPlot closed with Ctrl+C, returning to prompt...")
    plt.ioff()  # Disable interactive mode after closing


def interactive_experiment():
    """
    Runs the quantum experiment interactively with rerun and skip options.

    Users can choose default settings, manually enter parameters, or modify them after an experiment.
    Results are saved and optionally visualized with non-blocking plots, closable with Ctrl+C.
    """
    while True:
        print("\n🚀 Welcome to the Quantum Experiment Interactive Runner!")
        print("🔹 Choose an option:")
        print("🔄 Press 's' to skip and use default settings")
        print("🆕 Press 'n' to enter parameters manually")
        print("❌ Press 'q' to quit")

        choice = get_input("➡️ Your choice: ", "s", ["s", "n", "q"])

        # Initialize args with all required keys, using defaults as fallback
        args = {
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

        if choice == "s":
            print("\n⚡ Running with default configuration...\n")
            # Prompt for quick visualization choice
            viz_choice = get_input(
                "Show visualization? (p/plot, h/hypergraph, n/none) [p]: ",
                "p",
                ["p", "h", "n"],
            )
            args["visualization_type"] = (
                "plot"
                if viz_choice in ["p", "plot"]
                else "hypergraph" if viz_choice in ["h", "hypergraph"] else "none"
            )
            if args["visualization_type"] != "none":
                args["save_plot"] = (
                    input("Enter path to save (press Enter for display): ").strip()
                    or None
                )
                if args["visualization_type"] == "plot" and args["sim_mode"] == "qasm":
                    args["min_occurrences"] = int(
                        input(f"Minimum occurrences [0]: ") or 0
                    )
        elif choice == "n":
            print("\n🔹 Enter your experiment parameters below:\n")

            # Noise type with shortcuts and auto-correction
            noise_input = get_input(
                f"Enter noise type {VALID_NOISE_TYPES} (d/p/a/z/t/b) [{DEFAULT_NOISE_TYPE.lower()}]: ",
                DEFAULT_NOISE_TYPE.lower(),
            )
            args["noise_type"] = NOISE_SHORTCUTS.get(noise_input, noise_input.upper())

            # Visualization selection
            viz_choice = get_input(
                "\n🎨 Choose visualization type (p/plot, h/hypergraph, n/none): ",
                "n",
                ["p", "h", "n"],
            )
            args["visualization_type"] = (
                "plot"
                if viz_choice in ["p", "plot"]
                else "hypergraph" if viz_choice in ["h", "hypergraph"] else "none"
            )
            if args["visualization_type"] != "none":
                args["save_plot"] = (
                    input("Enter path to save plot (press Enter for display): ").strip()
                    or None
                )
                if args["visualization_type"] == "plot":
                    if args["sim_mode"] == "qasm":
                        args["min_occurrences"] = int(
                            input(f"Minimum occurrences [0]: ") or 0
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
            args["num_qubits"] = int(
                get_input(
                    f"Number of qubits [{DEFAULT_NUM_QUBITS}]: ",
                    str(DEFAULT_NUM_QUBITS),
                )
            )
            args["state_type"] = get_input(
                f"State type {VALID_STATE_TYPES} [{DEFAULT_STATE_TYPE.lower()}]: ",
                DEFAULT_STATE_TYPE.lower(),
                VALID_STATE_TYPES,
            ).upper()
            args["noise_enabled"] = get_input(
                f"Enable noise? (y/yes/t/true, n/no/f/false) [{str(DEFAULT_NOISE_ENABLED).lower()}]: ",
                str(DEFAULT_NOISE_ENABLED).lower(),
                ["y", "yes", "t", "true", "n", "no", "f", "false"],
            ) in ["y", "yes", "t", "true"]
            args["shots"] = int(
                get_input(f"Number of shots [{DEFAULT_SHOTS}]: ", str(DEFAULT_SHOTS))
            )
            args["sim_mode"] = get_input(
                f"Simulation mode (qasm/density) [{DEFAULT_SIM_MODE.lower()}]: ",
                DEFAULT_SIM_MODE.lower(),
                ["qasm", "density"],
            )

            # Optional parameters with confirmation
            if get_input("Set custom error rate? (y/n) [n]: ", "n", ["y", "n"]) == "y":
                args["error_rate"] = float(
                    get_input(
                        f"Error rate [{DEFAULT_ERROR_RATE}]: ", str(DEFAULT_ERROR_RATE)
                    )
                )
            if (
                args["noise_type"] == "PHASE_FLIP"
                and get_input(
                    "Set custom Z/I probabilities? (y/n) [n]: ", "n", ["y", "n"]
                )
                == "y"
            ):
                args["z_prob"] = float(
                    get_input("Z probability for PHASE_FLIP [0.5]: ", "0.5")
                )
                args["i_prob"] = float(
                    get_input("I probability for PHASE_FLIP [0.5]: ", "0.5")
                )
            if (
                args["noise_type"] == "THERMAL_RELAXATION"
                and get_input("Set custom T1/T2? (y/n) [n]: ", "n", ["y", "n"]) == "y"
            ):
                args["t1"] = (
                    float(get_input("T1 for THERMAL_RELAXATION (µs) [100]: ", "100"))
                    * 1e-6
                )
                args["t2"] = (
                    float(get_input("T2 for THERMAL_RELAXATION (µs) [80]: ", "80"))
                    * 1e-6
                )
            if (
                args["state_type"] == "CLUSTER"
                and get_input("Set custom lattice? (y/n) [n]: ", "n", ["y", "n"]) == "y"
            ):
                if "custom_params" not in args:
                    args["custom_params"] = {}
                lattice_type = get_input(
                    "Lattice type (1d/2d) [1d]: ", "1d", ["1d", "2d"]
                )
                args["custom_params"]["lattice"] = lattice_type
            if get_input("Set custom params? (y/n) [n]: ", "n", ["y", "n"]) == "y":
                custom_params_str = input(
                    "Enter custom params as JSON (press Enter for none): "
                ).strip()
                args["custom_params"] = (
                    json.loads(custom_params_str) if custom_params_str else None
                )

            # Validate Z/I probabilities for PHASE_FLIP
            if (
                args["noise_type"] == "PHASE_FLIP"
                and args["z_prob"] is not None
                and args["i_prob"] is not None
            ):
                if not (
                    0 <= args["z_prob"] <= 1
                    and 0 <= args["i_prob"] <= 1
                    and abs(args["z_prob"] + args["i_prob"] - 1) < 1e-10
                ):
                    print(
                        "⚠️ Z and I probabilities must sum to 1 and be between 0 and 1."
                    )
                    args["z_prob"], args["i_prob"] = None, None

            # Validate T1/T2 for THERMAL_RELAXATION
            if (
                args["noise_type"] == "THERMAL_RELAXATION"
                and args["t1"] is not None
                and args["t2"] is not None
            ):
                if args["t1"] <= 0 or args["t2"] <= 0 or args["t2"] > args["t1"]:
                    print(
                        "⚠️ T1 and T2 must be positive, with T2 <= T1 for realistic relaxation."
                    )
                    args["t1"], args["t2"] = None, None

            # Warn about 1-qubit noise limits
            if (
                args["noise_type"] in ["AMPLITUDE_DAMPING", "PHASE_DAMPING", "BIT_FLIP"]
                and args["num_qubits"] > 1
            ):
                print(
                    "⚠️ Note: This noise type applies only to 1-qubit gates, skipping multi-qubit gates (e.g., CNOTs)."
                )

        elif choice == "q":
            print("\n👋 Exiting Quantum Experiment Runner. Goodbye!")
            return
        else:
            print("⚠️ Invalid choice! Please enter s, n, or q.")
            continue

        # Run experiment
        logger.info(
            f"Starting experiment with {args['num_qubits']} qubits, {args['state_type']} state, "
            f"{'with' if args['noise_enabled'] else 'without'} {args['noise_type']} noise."
        )

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
            ]
        }

        try:
            result = run_experiment(**args_for_experiment)
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            print(f"⚠️ Experiment failed: {str(e)}")
            continue

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = (
            f"results/experiment_results_{args['num_qubits']}q_{args['state_type']}_"
            f"{args['noise_type']}_{args['sim_mode']}_{timestamp}.{'json' if args['sim_mode'] == 'qasm' else 'npy'}"
        )
        ExperimentUtils.save_results(result, filename)
        print(
            f"\n✅ Experiment completed successfully!\n📁 Results saved in `{filename}`"
        )

        # Handle visualization non-blockingly
        if args["visualization_type"] == "plot":
            if args["sim_mode"] == "qasm":
                if args["save_plot"]:
                    Visualizer.plot_histogram(
                        result["counts"],
                        state_type=args["state_type"],
                        noise_type=(
                            args["noise_type"] if args["noise_enabled"] else None
                        ),
                        noise_enabled=args["noise_enabled"],
                        save_path=args["save_plot"],
                        min_occurrences=args["min_occurrences"],
                    )
                else:
                    show_plot_nonblocking(
                        Visualizer.plot_histogram,
                        result["counts"],
                        state_type=args["state_type"],
                        noise_type=(
                            args["noise_type"] if args["noise_enabled"] else None
                        ),
                        noise_enabled=args["noise_enabled"],
                        min_occurrences=args["min_occurrences"],
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
                    )
                else:
                    show_plot_nonblocking(
                        Visualizer.plot_density_matrix,
                        result,
                        cmap="viridis",
                        show_real=args["show_real"],
                        show_imag=args["show_imag"],
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
                show_plot_nonblocking(
                    Visualizer.plot_hypergraph,
                    correlation_data,
                    state_type=args["state_type"],
                    noise_type=args["noise_type"] if args["noise_enabled"] else None,
                )

        # Ask for next action and handle reruns
        while True:
            params_str = format_params(args)
            print(f"\n🔄 Current parameters: {params_str}")
            next_choice = get_input(
                "\n➡️ Rerun? (r/same, n/new, q/quit): ", "r", ["r", "n", "q"]
            )
            if next_choice == "r":
                print("\n🔁 Rerunning with same parameters...\n")
                try:
                    result = run_experiment(**args_for_experiment)
                except Exception as e:
                    logger.error(f"Rerun failed: {str(e)}")
                    print(f"⚠️ Rerun failed: {str(e)}")
                    continue
                # Save rerun results with a suffix
                rerun_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                rerun_filename = (
                    f"results/experiment_results_{args['num_qubits']}q_{args['state_type']}_"
                    f"{args['noise_type']}_{args['sim_mode']}_{rerun_timestamp}_rerun.{'json' if args['sim_mode'] == 'qasm' else 'npy'}"
                )
                ExperimentUtils.save_results(result, rerun_filename)
                print(
                    f"\n✅ Rerun completed successfully!\n📁 Results saved in `{rerun_filename}`"
                )

                # Handle visualization for rerun
                if args["visualization_type"] == "plot":
                    if args["sim_mode"] == "qasm":
                        if args["save_plot"]:
                            Visualizer.plot_histogram(
                                result["counts"],
                                state_type=args["state_type"],
                                noise_type=(
                                    args["noise_type"]
                                    if args["noise_enabled"]
                                    else None
                                ),
                                noise_enabled=args["noise_enabled"],
                                save_path=args["save_plot"],
                                min_occurrences=args["min_occurrences"],
                            )
                        else:
                            show_plot_nonblocking(
                                Visualizer.plot_histogram,
                                result["counts"],
                                state_type=args["state_type"],
                                noise_type=(
                                    args["noise_type"]
                                    if args["noise_enabled"]
                                    else None
                                ),
                                noise_enabled=args["noise_enabled"],
                                min_occurrences=args["min_occurrences"],
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
                            )
                        else:
                            show_plot_nonblocking(
                                Visualizer.plot_density_matrix,
                                result,
                                cmap="viridis",
                                show_real=args["show_real"],
                                show_imag=args["show_imag"],
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
                            noise_type=(
                                args["noise_type"] if args["noise_enabled"] else None
                            ),
                            save_path=args["save_plot"],
                        )
                    else:
                        show_plot_nonblocking(
                            Visualizer.plot_hypergraph,
                            correlation_data,
                            state_type=args["state_type"],
                            noise_type=(
                                args["noise_type"] if args["noise_enabled"] else None
                            ),
                        )
                continue  # Back to rerun prompt
            elif next_choice == "n":
                print("\n🆕 Restarting parameter selection...\n")
                return interactive_experiment()
            else:  # 'q'
                print("\n👋 Exiting Quantum Experiment Runner. Goodbye!")
                return


if __name__ == "__main__":
    interactive_experiment()
