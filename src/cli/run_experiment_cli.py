#!/usr/bin/env python3

"""
CLI script to run quantum experiments, designed for extensibility and research integration.

This script provides a non-interactive command-line interface to execute quantum experiments,
integrating with a scalable "quantum experimental lab." Supports new noise models, states,
and research features (e.g., hypergraph correlations, fluid dynamics in Hilbert space).

Key features:
- Configurable CLI for all experiment parameters, including new noise and state options.
- Dynamic validation pulling from noise models and state preparation modules.
- Logging and results saving for research analysis (e.g., hypergraphs, density matrices).
- Optional visualization via `--plot` or `--plot_hypergraph` for immediate feedback.
- Extensible for custom params, config files, or new backends (e.g., IBM hardware).
- Handles Qiskit deprecation warnings and density mode errors.

Example usage:
    python scripts/run_experiment.py --num_qubits 3 --state_type GHZ --noise_type DEPOLARIZING \
        --error_rate 0.1 --plot histogram.png
"""

import numpy as np
import warnings
from typing import Optional
from qiskit.quantum_info import DensityMatrix
from src.run_experiment import run_experiment
from src.utils import ExperimentUtils
from src.visualization import Visualizer

# Suppress Qiskit deprecation warnings for cleaner logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logger
logger = ExperimentUtils.setup_logger()


def main():
    """
    Runs the quantum experiment from the CLI, with logging, validation, and optional visualization.

    Parses command-line arguments, validates inputs, executes the experiment, saves results,
    and optionally plots outcomes for research (e.g., hypergraph correlations).

    Raises:
        ValueError: If inputs are invalid or inconsistent.
    """
    # Parse CLI args using utils
    args = ExperimentUtils.parse_args()

    logger.info("Validating inputs...")
    ExperimentUtils.validate_inputs(
        num_qubits=args.num_qubits,
        state_type=args.state_type,
        noise_type=args.noise_type,
        sim_mode=args.sim_mode,
        angle=getattr(args, "angle", None),
        error_rate=getattr(args, "error_rate", None),
        z_prob=getattr(args, "z_prob", None),
        i_prob=getattr(args, "i_prob", None),
        t1=getattr(args, "t1", None),
        t2=getattr(args, "t2", None),
    )

    logger.info(
        f"Starting experiment with {args.num_qubits} qubits, {args.state_type} state, "
        f"{'with' if args.noise_enabled else 'without'} {args.noise_type} noise."
    )

    # Run experiment with all params, including new noise and state options
    try:
        result = run_experiment(
            num_qubits=args.num_qubits,
            state_type=args.state_type,
            noise_type=args.noise_type,
            noise_enabled=args.noise_enabled,
            shots=args.shots,
            sim_mode=args.sim_mode,
            error_rate=args.error_rate,
            z_prob=args.z_prob,
            i_prob=args.i_prob,
            t1=args.t1,
            t2=args.t2,
            custom_params=args.custom_params,
        )
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

    # Save results, handling both counts (JSON) and density matrices (NumPy)
    filename = (
        "cli_experiment_results.json"
        if args.sim_mode == "qasm"
        else "cli_experiment_results.npy"
    )
    ExperimentUtils.save_results(result, filename)
    logger.info(f"Experiment complete! Results saved to {filename}")

    # Optional visualization
    if hasattr(args, "plot") and args.plot:
        if args.sim_mode == "qasm":
            Visualizer.plot_histogram(
                result["counts"],  # Access counts from dict
                state_type=args.state_type,
                noise_type=args.noise_type if args.noise_enabled else None,
                noise_enabled=args.noise_enabled,
                save_path=args.plot,
                min_occurrences=1,  # Filter rare outcomes
            )
        else:
            Visualizer.plot_density_matrix(
                result,
                show_real=False,
                show_imag=False,
                save_path=args.plot,
            )
        logger.info(f"Plotted results to {args.plot}")

    if hasattr(args, "plot_hypergraph") and args.plot_hypergraph:
        correlation_data = (
            result["counts"]
            if args.sim_mode == "qasm"
            else (
                {"density": np.abs(result.data).tolist()}
                if isinstance(result, DensityMatrix)
                else result
            )  # Fallback for unexpected types
        )
        Visualizer.plot_hypergraph(
            correlation_data,
            state_type=args.state_type,
            noise_type=args.noise_type if args.noise_enabled else None,
            save_path=args.plot_hypergraph,
        )
        logger.info(f"Plotted hypergraph to {args.plot_hypergraph}")

    print("\n✅ Experiment completed successfully!")
    print(f"📁 Results saved in `{filename}`")


if __name__ == "__main__":
    main()
