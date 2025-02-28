# src/scripts/run_experiment_cli.py

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
- Optional visualization via `--plot` or `--plot-hypergraph` for immediate feedback.
- Support for time-stepped simulations to analyze decoherence dynamics with multiple noise parameters.
- Extensible for custom params, config files, or new backends (e.g., IBM hardware).
- Handles Qiskit deprecation warnings and density mode errors.

Example usage:
    python scripts/run_experiment.py --num_qubits 3 --state_type GHZ --noise_type DEPOLARIZING \
        --error_rate 0.1 --plot histogram.png
    python scripts/run_experiment.py --num_qubits 3 --state_type GHZ --noise_type DEPOLARIZING \
        --plot-hypergraph hypergraph.png --noise-stepped --noise-start 0.0 --noise-end 0.5 --noise-steps 10
"""

import click
import numpy as np
import warnings
from typing import Optional, List, Union, Dict
from tqdm import tqdm
from qiskit.quantum_info import DensityMatrix
from src.run_experiment import run_experiment
from src.utils import ExperimentUtils
from src.visualization import Visualizer
from src.visualization.hypergraph import plot_hypergraph
from src.config.constants import VALID_STATE_TYPES, VALID_NOISE_TYPES

# Suppress Qiskit deprecation warnings for cleaner logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logger
logger = ExperimentUtils.setup_logger()


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
@click.option("--plot", type=str, help="Path to save histogram or density matrix plot")
@click.option("--plot-hypergraph", type=str, help="Path to save hypergraph plot")
# Hypergraph-specific options
@click.option(
    "--hypergraph-max-order",
    type=int,
    default=2,
    help="Maximum order of correlations for hypergraph (2-3)",
)
@click.option(
    "--hypergraph-threshold",
    type=float,
    help="Correlation threshold for hypergraph edges",
)
@click.option(
    "--hypergraph-symmetry/--no-hypergraph-symmetry",
    default=False,
    help="Perform symmetry analysis for hypergraph",
)
@click.option(
    "--hypergraph-transitions/--no-hypergraph-transitions",
    default=False,
    help="Plot error transition graph for hypergraph (requires --noise-stepped)",
)
# Time-stepped simulation options
@click.option(
    "--noise-stepped/--no-noise-stepped",
    default=False,
    help="Run simulation with stepped noise levels",
)
@click.option(
    "--noise-start", type=float, default=0.0, help="Starting noise error rate"
)
@click.option("--noise-end", type=float, default=0.5, help="Ending noise error rate")
@click.option("--noise-steps", type=int, default=10, help="Number of noise steps")
# Stepped options for other noise parameters
@click.option(
    "--z-prob-start", type=float, help="Starting Z probability for PHASE_FLIP noise"
)
@click.option(
    "--z-prob-end", type=float, help="Ending Z probability for PHASE_FLIP noise"
)
@click.option(
    "--i-prob-start", type=float, help="Starting I probability for PHASE_FLIP noise"
)
@click.option(
    "--i-prob-end", type=float, help="Ending I probability for PHASE_FLIP noise"
)
@click.option(
    "--t1-start",
    type=float,
    help="Starting T1 relaxation time (µs) for THERMAL_RELAXATION noise",
)
@click.option(
    "--t1-end",
    type=float,
    help="Ending T1 relaxation time (µs) for THERMAL_RELAXATION noise",
)
@click.option(
    "--t2-start",
    type=float,
    help="Starting T2 dephasing time (µs) for THERMAL_RELAXATION noise",
)
@click.option(
    "--t2-end",
    type=float,
    help="Ending T2 dephasing time (µs) for THERMAL_RELAXATION noise",
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
    plot: Optional[str],
    plot_hypergraph: Optional[str],
    hypergraph_max_order: int,
    hypergraph_threshold: Optional[float],
    hypergraph_symmetry: bool,
    hypergraph_transitions: bool,
    noise_stepped: bool,
    noise_start: float,
    noise_end: float,
    noise_steps: int,
    z_prob_start: Optional[float],
    z_prob_end: Optional[float],
    i_prob_start: Optional[float],
    i_prob_end: Optional[float],
    t1_start: Optional[float],
    t1_end: Optional[float],
    t2_start: Optional[float],
    t2_end: Optional[float],
):
    """
    Runs the quantum experiment from the CLI, with logging, validation, and optional visualization.

    Parses command-line arguments, validates inputs, executes the experiment, saves results,
    and optionally plots outcomes for research (e.g., hypergraph correlations).

    Raises:
        ValueError: If inputs are invalid or inconsistent.
    """
    # Enforce dependency: --hypergraph-transitions requires --noise-stepped
    if hypergraph_transitions and not noise_stepped:
        raise click.UsageError(
            "Error: --hypergraph-transitions requires --noise-stepped to be enabled"
        )

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

    # Run experiment
    try:
        if noise_stepped and noise_enabled:
            # Generate stepped parameters
            error_rates = (
                np.linspace(noise_start, noise_end, noise_steps)
                if noise_start is not None and noise_end is not None
                else np.array([args.error_rate] * noise_steps)
            )
            z_probs = (
                np.linspace(z_prob_start, z_prob_end, noise_steps)
                if z_prob_start is not None and z_prob_end is not None
                else np.array(
                    [args.z_prob if args.z_prob is not None else 0.0] * noise_steps
                )
            )
            i_probs = (
                np.linspace(i_prob_start, i_prob_end, noise_steps)
                if i_prob_start is not None and i_prob_end is not None
                else np.array(
                    [args.i_prob if args.i_prob is not None else 0.0] * noise_steps
                )
            )
            t1_values = (
                np.linspace(t1_start, t1_end, noise_steps)
                if t1_start is not None and t1_end is not None
                else np.array([args.t1 if args.t1 is not None else 100.0] * noise_steps)
            )
            t2_values = (
                np.linspace(t2_start, t2_end, noise_steps)
                if t2_start is not None and t2_end is not None
                else np.array([args.t2 if args.t2 is not None else 80.0] * noise_steps)
            )

            results = []
            for idx in tqdm(range(noise_steps), desc="Running stepped simulations"):
                logger.info(
                    f"Running step {idx+1}/{noise_steps}: error_rate={error_rates[idx]:.3f}, "
                    f"z_prob={z_probs[idx]:.3f}, i_prob={i_probs[idx]:.3f}, "
                    f"t1={t1_values[idx]:.3f}, t2={t2_values[idx]:.3f}"
                )
                result = run_experiment(
                    num_qubits=args.num_qubits,
                    state_type=args.state_type,
                    noise_type=args.noise_type,
                    noise_enabled=args.noise_enabled,
                    shots=args.shots,
                    sim_mode=args.sim_mode,
                    error_rate=error_rates[idx],
                    z_prob=z_probs[idx] if z_probs[idx] != 0.0 else None,
                    i_prob=i_probs[idx] if i_probs[idx] != 0.0 else None,
                    t1=t1_values[idx] if t1_values[idx] != 100.0 else None,
                    t2=t2_values[idx] if t2_values[idx] != 80.0 else None,
                    custom_params=args.custom_params,
                )
                results.append(result)
            time_steps = (
                error_rates.tolist()
            )  # Use error rates as time steps for visualization
        else:
            # Single run with specified parameters
            results = run_experiment(
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
            time_steps = None
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

    # Save results, handling both counts (JSON) and density matrices (NumPy)
    filename = (
        "cli_experiment_results.json"
        if args.sim_mode == "qasm"
        else "cli_experiment_results.npy"
    )
    # If stepped, save the last result; otherwise, save the single result
    ExperimentUtils.save_results(
        results[-1] if noise_stepped and noise_enabled else results, filename
    )
    logger.info(f"Experiment complete! Results saved to {filename}")

    # Optional visualization (plot)
    if plot:
        # Use the last result for static plots if stepped
        result_to_plot = results[-1] if noise_stepped and noise_enabled else results
        if args.sim_mode == "qasm":
            Visualizer.plot_histogram(
                result_to_plot["counts"],
                state_type=args.state_type,
                noise_type=args.noise_type if args.noise_enabled else None,
                noise_enabled=args.noise_enabled,
                save_path=args.plot,
                min_occurrences=1,
            )
        else:
            Visualizer.plot_density_matrix(
                result_to_plot,
                show_real=False,
                show_imag=False,
                save_path=args.plot,
            )
        logger.info(f"Plotted results to {args.plot}")

    # Optional hypergraph visualization
    if plot_hypergraph:
        # Prepare correlation data
        if noise_stepped and noise_enabled:
            correlation_data = [
                (
                    result["counts"]
                    if args.sim_mode == "qasm"
                    else (
                        {"density": np.abs(result.data).tolist()}
                        if isinstance(result, DensityMatrix)
                        else result
                    )
                )
                for result in results
            ]
        else:
            correlation_data = (
                results["counts"]
                if args.sim_mode == "qasm"
                else (
                    {"density": np.abs(results.data).tolist()}
                    if isinstance(results, DensityMatrix)
                    else results
                )
            )
            time_steps = None

        # Configure hypergraph settings
        hypergraph_config = {
            "max_order": hypergraph_max_order,
            "threshold": hypergraph_threshold,
            "symmetry_analysis": hypergraph_symmetry,
            "plot_transitions": hypergraph_transitions,
        }
        plot_hypergraph(
            correlation_data,
            state_type=args.state_type,
            noise_type=args.noise_type if args.noise_enabled else None,
            save_path=args.plot_hypergraph,
            time_steps=time_steps,
            config=hypergraph_config,
        )
        logger.info(f"Plotted hypergraph to {args.plot_hypergraph}")

    print("\n✅ Experiment completed successfully!")
    print(f"📁 Results saved in `{filename}`")


if __name__ == "__main__":
    main()
