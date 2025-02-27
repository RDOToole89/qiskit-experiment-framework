# src/quantum_experiment/utils.py

"""
Utility functions for quantum experiments, designed for extensibility and research integration.

This module provides tools for CLI parsing, logging, input validation, and results management,
supporting a scalable "quantum experimental lab." Integrates with new noise models, states,
and research features (e.g., hypergraph correlations, fluid dynamics in Hilbert space).

Key features:
- Configurable CLI parsing for all experiment parameters, including new noise and state options.
- Enhanced logging with noise/state-specific metadata for hypergraph or density matrix analysis.
- Flexible input validation, pulling valid options from config or other modules.
- Results management supporting JSON (counts) and NumPy/DensityMatrix (density mode) for research.
- Extensible for custom params, config files, or new utilities (e.g., hypergraph data).

Example usage:
    args = parse_args()
    logger = setup_logger()
    validate_inputs(args.num_qubits, args.state_type, args.noise_type, args.sim_mode)
    save_results({"counts": counts}, "experiment_results.json")
"""

import os
import json
import logging
import argparse
from datetime import datetime
import numpy as np
from qiskit.quantum_info import DensityMatrix
from typing import Optional, Dict, Union, Any
from src.config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)
from src.noise_models import NOISE_CLASSES
from src.state_preparation import STATE_CLASSES

# Configure logger for utility-specific debugging
logger = logging.getLogger("QuantumExperiment.Utils")


class ExperimentUtils:
    """
    Utility class for quantum experiment management, providing modular and extensible tools.

    Attributes:
        None (static methods for now, but extensible for instance-specific behavior).
    """

    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parses command-line arguments for experiment execution, including new noise and state params.

        Supports configurable noise levels (error_rate, z_prob, etc.) and custom parameters
        for research (e.g., hypergraph data).

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        Example:
            python scripts/run_experiment.py --num_qubits 3 --noise_type DEPOLARIZING --error_rate 0.15
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

        # Validate PHASE_FLIP probabilities if provided
        if args.noise_type == "PHASE_FLIP" and (
            args.z_prob is not None or args.i_prob is not None
        ):
            if args.z_prob is None or args.i_prob is None:
                raise ValueError(
                    "Both --z_prob and --i_prob must be provided for PHASE_FLIP noise."
                )
            if not (
                0 <= args.z_prob <= 1
                and 0 <= args.i_prob <= 1
                and abs(args.z_prob + args.i_prob - 1) < 1e-10
            ):
                raise ValueError(
                    "Z and I probabilities for PHASE_FLIP must sum to 1 and be between 0 and 1."
                )

        # Parse custom_params as JSON if provided
        if args.custom_params:
            try:
                args.custom_params = json.loads(args.custom_params)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for --custom_params.")

        return args

    @staticmethod
    def setup_logger() -> logging.Logger:
        """
        Configures logging with automatic log file creation in logs/ directory.

        Supports noise and state-specific logging for hypergraph or fluid dynamics analysis.

        Returns:
            logging.Logger: Configured logger instance.
        """
        # Ensure `logs/` directory exists
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Create a timestamped log file
        log_filename = os.path.join(
            log_dir, f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )

        # Setup logging configuration with detailed format
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),  # Save logs to file
                logging.StreamHandler(),  # Print logs to console
            ],
        )

        logger = logging.getLogger("QuantumExperiment")
        logger.debug("Logger configured for quantum experiment utilities")
        return logger

    @staticmethod
    def validate_inputs(
        num_qubits: int,
        state_type: str,
        noise_type: str,
        sim_mode: str,
        angle: Optional[float] = None,
        error_rate: Optional[float] = None,
        z_prob: Optional[float] = None,
        i_prob: Optional[float] = None,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
    ) -> None:
        """
        Validates input parameters to ensure correctness, pulling valid options dynamically.

        Checks new noise and state params (e.g., angle, error_rate) for consistency with
        research goals (hypergraphs, fluid dynamics).

        Args:
            num_qubits (int): Number of qubits (minimum 1).
            state_type (str): Quantum state type (e.g., "GHZ").
            noise_type (str): Noise type (e.g., "DEPOLARIZING").
            sim_mode (str): Simulation mode ("qasm" or "density").
            angle (float, optional): Angle for CLUSTER state (radians).
            error_rate (float, optional): Base error rate for noise.
            z_prob (float, optional): Z probability for PHASE_FLIP noise.
            i_prob (float, optional): I probability for PHASE_FLIP noise.
            t1 (float, optional): T1 relaxation time for THERMAL_RELAXATION.
            t2 (float, optional): T2 dephasing time for THERMAL_RELAXATION.

        Raises:
            ValueError: If parameters are invalid or inconsistent.
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")

        valid_states = list(STATE_CLASSES.keys())
        if state_type not in valid_states:
            raise ValueError(
                f"Invalid state type: {state_type}. Choose from {valid_states}"
            )

        valid_noises = list(NOISE_CLASSES.keys())
        if noise_type not in valid_noises:
            raise ValueError(
                f"Invalid noise type: {noise_type}. Choose from {valid_noises}"
            )

        if sim_mode not in ["qasm", "density"]:
            raise ValueError(
                f"Invalid simulation mode: {sim_mode}. Choose from ['qasm', 'density']"
            )

        # Validate CLUSTER angle if state_type is CLUSTER
        if state_type == "CLUSTER" and angle is not None:
            if not (0 <= angle <= 2 * np.pi):
                raise ValueError(
                    "Angle for CLUSTER state must be between 0 and 2π radians."
                )

        # Validate noise parameters
        if error_rate is not None and not (0 <= error_rate <= 1):
            raise ValueError("Error rate must be between 0 and 1.")

        if noise_type == "PHASE_FLIP" and (z_prob is not None or i_prob is not None):
            if z_prob is None or i_prob is None:
                raise ValueError(
                    "Both z_prob and i_prob must be provided for PHASE_FLIP noise."
                )
            if not (
                0 <= z_prob <= 1
                and 0 <= i_prob <= 1
                and abs(z_prob + i_prob - 1) < 1e-10
            ):
                raise ValueError(
                    "Z and I probabilities for PHASE_FLIP must sum to 1 and be between 0 and 1."
                )

        if noise_type == "THERMAL_RELAXATION" and (t1 is not None or t2 is not None):
            if t1 is None or t2 is None:
                raise ValueError(
                    "Both t1 and t2 must be provided for THERMAL_RELAXATION noise."
                )
            if t1 <= 0 or t2 <= 0 or t2 > t1:
                raise ValueError(
                    "T1 and T2 must be positive, with T2 <= T1 for realistic relaxation."
                )

        logger.debug(
            f"Validated inputs: num_qubits={num_qubits}, state_type={state_type}, "
            f"noise_type={noise_type}, sim_mode={sim_mode}"
        )

    @staticmethod
    def save_results(
        result: Union[Dict, DensityMatrix], filename: str = "experiment_results.json"
    ) -> None:
        """
        Saves experiment results to a JSON or NumPy file inside the results directory.

        Supports both counts (dict) and density matrices (DensityMatrix) for research analysis
        (e.g., hypergraph correlations, fluid dynamics).

        Args:
            result (Union[Dict, DensityMatrix]): Experiment results (counts or density matrix).
            filename (str): Output filename (default "experiment_results.json" or ".npy" for density).

        Raises:
            ValueError: If result type is unsupported.
        """
        RESULTS_DIR = "results"
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        # Construct the full path
        full_filename = os.path.join(RESULTS_DIR, filename)

        if isinstance(result, dict):  # Counts from qasm mode
            os.makedirs(
                os.path.dirname(full_filename), exist_ok=True
            )  # Ensure directory exists
            with open(full_filename, "w") as f:
                json.dump(result, f, indent=4)
            logger.info(f"Saved qasm results to {full_filename}")
        elif isinstance(result, DensityMatrix):  # Density matrix from density mode
            np_filename = full_filename.rsplit(".", 1)[0] + ".npy"
            np.save(np_filename, np.array(result.data, dtype=complex))
            logger.info(f"Saved density matrix results to {np_filename}")
        else:
            raise ValueError(
                f"Unsupported result type: {type(result)}. Expected Dict or DensityMatrix."
            )

        print(
            f"✅ Results saved to {full_filename if isinstance(result, dict) else np_filename}"
        )

    @staticmethod
    def load_results(
        filename: str = "experiment_results.json",
    ) -> Union[Dict, DensityMatrix]:
        """
        Loads experiment results from a JSON or NumPy file.

        Supports counts (JSON) and density matrices (NumPy) for research analysis
        (e.g., hypergraph correlations, fluid dynamics).

        Args:
            filename (str): Input filename (default "experiment_results.json" or ".npy").

        Returns:
            Union[Dict, DensityMatrix]: Loaded results (counts or density matrix).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is unsupported.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Error: {filename} not found.")

        if filename.endswith(".json"):
            with open(filename, "r") as f:
                return json.load(f)
        elif filename.endswith(".npy"):
            data = np.load(filename, allow_pickle=True)
            return DensityMatrix(data)
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .json or .npy.")

    @staticmethod
    def load_config(config_file: str = "config.json") -> Dict:
        """
        Loads experiment configuration from a JSON file for batch runs or defaults.

        Supports noise rates, state params, and custom settings for research (e.g., hypergraphs).

        Args:
            config_file (str): Path to JSON config file (default "config.json").

        Returns:
            Dict: Configuration parameters.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file is invalid.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Error: {config_file} not found.")

        with open(config_file, "r") as f:
            config = json.load(f)
        logger.debug(f"Loaded configuration from {config_file}: {config}")
        return config
