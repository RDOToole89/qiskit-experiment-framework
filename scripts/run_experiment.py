# src/scripts/run_experiment.py
#!/usr/bin/env python3

"""
CLI script to run quantum experiments.
"""

import sys
import argparse
import logging
from quantum_experiment.run_experiment import run_experiment
from quantum_experiment.utils import setup_logger, validate_inputs, save_results
from quantum_experiment.config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)

# Configure logger
logger = setup_logger()


def parse_cli_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a quantum experiment.")

    parser.add_argument(
        "--num_qubits",
        type=int,
        default=DEFAULT_NUM_QUBITS,
        help=f"Number of qubits (default: {DEFAULT_NUM_QUBITS})",
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default=DEFAULT_STATE_TYPE,
        choices=["GHZ", "W", "G-CRY"],
        help=f"Quantum state type (default: {DEFAULT_STATE_TYPE})",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default=DEFAULT_NOISE_TYPE,
        choices=["DEPOLARIZING", "PHASE_FLIP", "AMPLITUDE_DAMPING", "PHASE_DAMPING"],
        help=f"Type of noise (default: {DEFAULT_NOISE_TYPE})",
    )
    parser.add_argument(
        "--noise_enabled",
        action="store_true",
        default=DEFAULT_NOISE_ENABLED,
        help=f"Enable noise? (default: {DEFAULT_NOISE_ENABLED})",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"Number of shots (default: {DEFAULT_SHOTS})",
    )
    parser.add_argument(
        "--sim_mode",
        type=str,
        default=DEFAULT_SIM_MODE,
        choices=["qasm", "density"],
        help=f"Simulation mode (default: {DEFAULT_SIM_MODE})",
    )

    return parser.parse_args()


def main():
    """Runs the quantum experiment from the CLI."""
    args = parse_cli_args()

    logger.info("Validating inputs...")
    validate_inputs(
        args.num_qubits,
        args.state_type,
        args.noise_type,
        args.sim_mode,
        valid_states=["GHZ", "W", "G-CRY"],
        valid_noises=[
            "DEPOLARIZING",
            "PHASE_FLIP",
            "AMPLITUDE_DAMPING",
            "PHASE_DAMPING",
        ],
        valid_modes=["qasm", "density"],
    )

    logger.info(
        f"Starting experiment with {args.num_qubits} qubits, {args.state_type} state, "
        f"{'with' if args.noise_enabled else 'without'} {args.noise_type} noise."
    )

    result = run_experiment(
        num_qubits=args.num_qubits,
        state_type=args.state_type,
        noise_type=args.noise_type,
        noise_enabled=args.noise_enabled,
        shots=args.shots,
        sim_mode=args.sim_mode,
    )

    # Save results
    save_results(result.to_dict(), "cli_experiment_results.json")
    logger.info("Experiment complete! Results saved.")

    print("\n✅ Experiment completed successfully!")
    print("📁 Results saved in `cli_experiment_results.json`")


if __name__ == "__main__":
    main()
