# src/quantum_experiment/utils.py

"""
Utility functions for quantum experiments.
"""

import os
import json
import logging
import argparse
from datetime import datetime


def parse_args():
    """Parses command-line arguments for experiment execution."""
    parser = argparse.ArgumentParser(description="Run a quantum experiment.")

    parser.add_argument(
        "--num_qubits", type=int, default=3, help="Number of qubits (default: 3)"
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default="GHZ",
        choices=["GHZ", "W", "G-CRY"],
        help="Quantum state type (default: GHZ)",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="DEPOLARIZING",
        choices=["DEPOLARIZING", "PHASE_FLIP", "AMPLITUDE_DAMPING", "PHASE_DAMPING"],
        help="Type of noise (default: DEPOLARIZING)",
    )
    parser.add_argument(
        "--noise_enabled",
        action="store_true",
        default=True,
        help="Enable noise? (default: True)",
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Number of shots (default: 1024)"
    )
    parser.add_argument(
        "--sim_mode",
        type=str,
        default="qasm",
        choices=["qasm", "density"],
        help="Simulation mode (default: qasm)",
    )

    return parser.parse_args()


def setup_logger():
    """Configures logging with automatic log file creation in logs/ directory."""

    # ✅ Ensure `logs/` directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Create a timestamped log file
    log_filename = os.path.join(
        log_dir, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    # ✅ Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  # ✅ Save logs to file
            logging.StreamHandler(),  # ✅ Print logs to console
        ],
    )

    logger = logging.getLogger("QuantumExperiment")
    return logger


def validate_inputs(
    num_qubits,
    state_type,
    noise_type,
    sim_mode,
    valid_states=["GHZ", "W", "G-CRY"],
    valid_noises=["DEPOLARIZING", "PHASE_FLIP", "AMPLITUDE_DAMPING", "PHASE_DAMPING"],
    valid_modes=["qasm", "density"],
):
    """Validates input parameters to ensure correctness."""
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")

    if state_type not in valid_states:
        raise ValueError(
            f"Invalid state type: {state_type}. Choose from {valid_states}"
        )

    if noise_type not in valid_noises:
        raise ValueError(
            f"Invalid noise type: {noise_type}. Choose from {valid_noises}"
        )

    if sim_mode not in valid_modes:
        raise ValueError(
            f"Invalid simulation mode: {sim_mode}. Choose from {valid_modes}"
        )


def save_results(result, filename="experiment_results.json"):
    """Saves experiment results to a JSON file inside the results directory."""

    # Ensure the results directory exists
    results_dir = "results"
    os.makedirs(
        results_dir, exist_ok=True
    )  # ✅ Creates the directory if it doesn't exist

    # Ensure filename doesn't contain duplicate "results/" prefix
    filename = filename.lstrip("results/")

    # Construct the full path
    full_filename = os.path.join(results_dir, filename)

    # Save JSON file
    with open(full_filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Results saved to {full_filename}")


def load_results(filename="experiment_results.json"):
    """Loads experiment results from a JSON file."""

    # ✅ Check if file exists before loading
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found.")

    with open(filename, "r") as f:
        return json.load(f)
