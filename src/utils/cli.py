import argparse
import json
import numpy as np
from src.config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)
from src.state_preparation import STATE_CLASSES
from src.noise_models import NOISE_CLASSES

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for experiment execution.
    """
    parser = argparse.ArgumentParser(description="Run a quantum experiment.")

    parser.add_argument(
        "--num_qubits",
        type=int,
        default=DEFAULT_NUM_QUBITS,
        help=f"Number of qubits (default: {DEFAULT_NUM_QUBITS}, minimum 1)",
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default=DEFAULT_STATE_TYPE,
        choices=list(STATE_CLASSES.keys()),
        help=f"Quantum state type (default: {DEFAULT_STATE_TYPE})",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default=DEFAULT_NOISE_TYPE,
        choices=list(NOISE_CLASSES.keys()),
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
    parser.add_argument(
        "--error_rate",
        type=float,
        default=None,
        help="Base error rate for noise models (overrides default if provided)",
    )
    parser.add_argument(
        "--z_prob",
        type=float,
        default=None,
        help="Z probability for PHASE_FLIP noise (requires --i_prob, sums to 1)",
    )
    parser.add_argument(
        "--i_prob",
        type=float,
        default=None,
        help="I probability for PHASE_FLIP noise (requires --z_prob, sums to 1)",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help="T1 relaxation time for THERMAL_RELAXATION noise (seconds)",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=None,
        help="T2 dephasing time for THERMAL_RELAXATION noise (seconds)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Angle for CLUSTER state (radians, default π/3 if not specified)",
    )
    parser.add_argument(
        "--custom_params",
        type=str,
        default=None,
        help='Custom parameters as JSON string (e.g., \'{"hypergraph_data": {"correlations": [...]}}\')',
    )
    parser.add_argument(
        "--plot_hypergraph",
        type=str,
        default=None,
        help="Path to save the hypergraph plot (e.g., 'hypergraph.png').",
    )

    args = parser.parse_args()

    # Validate PHASE_FLIP probabilities if provided.
    if args.noise_type == "PHASE_FLIP" and (args.z_prob is not None or args.i_prob is not None):
        if args.z_prob is None or args.i_prob is None:
            raise ValueError("Both --z_prob and --i_prob must be provided for PHASE_FLIP noise.")
        if not (
            0 <= args.z_prob <= 1
            and 0 <= args.i_prob <= 1
            and abs(args.z_prob + args.i_prob - 1) < 1e-10
        ):
            raise ValueError("Z and I probabilities for PHASE_FLIP must sum to 1 and be between 0 and 1.")

    # Parse custom_params as JSON if provided.
    if args.custom_params:
        try:
            args.custom_params = json.loads(args.custom_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for --custom_params.")

    return args
