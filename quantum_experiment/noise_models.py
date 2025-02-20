#!/usr/bin/env python3
# src/quantum_experiment/noise_models.py

"""
Noise models for quantum experiments, designed for maximum extensibility and integration with research goals
(such as hypergraph correlations, fluid dynamics, and structured decoherence patterns).

This module provides:
- A flexible, object-oriented noise framework for easy extension (e.g., new noise types).
- Configurable error rates per noise type, adjustable via CLI or config files.
- Scalable noise application for 1+ qubit gates, supporting multi-qubit systems.
- New noise models (e.g., thermal relaxation, bit flip) to study entanglement constraints.
- Logging of noise effects on correlations for hypergraph or symplectic analysis.

Key features:
- Noise classes encapsulate behavior, making it easy to add custom models.
- Configurable noise levels allow tuning for realism or research (e.g., studying parity preservation).
- Supports Qiskit’s noise model for gates (1-qubit: id/u1/u2/u3, 2+-qubit: cx and beyond).
- Tracks correlation impacts for hypergraph mapping or fluid dynamics insights.

Example usage:
    noise_model = create_noise_model("DEPOLARIZING", num_qubits=3, error_rate=0.1)
    # Or configure PHASE_FLIP with custom Z/I probabilities:
    noise_model = create_noise_model("PHASE_FLIP", num_qubits=1, z_prob=0.2, i_prob=0.8)
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
)
from typing import Dict, Optional, Union
import logging
from datetime import datetime

# Configure logger for noise-specific debugging
logger = logging.getLogger("QuantumExperiment.NoiseModels")

# Default noise parameters (can be overridden via config or CLI)
DEFAULT_ERROR_RATE = 0.1  # Base error rate for most noise types
DEFAULT_T1 = 100e-6  # Thermal relaxation time (100 microseconds, typical for qubits)
DEFAULT_T2 = 80e-6  # Dephasing time (80 microseconds, typical for qubits)


class BaseNoise:
    """
    Base class for all noise models, providing a template for noise application.

    Attributes:
        error_rate (float): Probability of error occurrence (configurable).
        num_qubits (int): Number of qubits affected by the noise.
    """

    def __init__(self, error_rate: float = DEFAULT_ERROR_RATE, num_qubits: int = 1):
        self.error_rate = error_rate
        self.num_qubits = num_qubits

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        """
        Applies noise to the specified gates in the noise model.

        Args:
            noise_model (NoiseModel): Qiskit noise model to modify.
            gate_list (list): List of gate names (e.g., ['id', 'u1']) to apply noise to.
        """
        raise NotImplementedError("Subclasses must implement apply()")


class DepolarizingNoise(BaseNoise):
    """
    Depolarizing noise, modeling random bit flips across qubits.

    Scales error rate with qubit count, suitable for studying random decoherence patterns.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        """
        Applies depolarizing noise correctly based on gate size.

        Args:
            noise_model (NoiseModel): The Qiskit noise model.
            gate_list (list): List of gates to apply noise to.
        """
        for gate in gate_list:
            if gate in ["id", "u1", "u2", "u3"]:  # 1-qubit gates
                noise = depolarizing_error(self.error_rate, 1)
                noise_model.add_all_qubit_quantum_error(noise, [gate])
            elif gate in ["cx"]:  # 2-qubit gates
                noise = depolarizing_error(self.error_rate, 2)
                noise_model.add_all_qubit_quantum_error(noise, [gate])
            elif "mct" in gate:  # Multi-controlled Toffoli (3+ qubits)
                qubit_count = int(
                    gate.split("_")[1]
                )  # Extract qubit count from gate name
                noise = depolarizing_error(self.error_rate, qubit_count)
                noise_model.add_all_qubit_quantum_error(noise, [gate])

        logger.debug(
            f"Applied depolarizing noise (rate={self.error_rate}) to {gate_list}"
        )


class PhaseFlipNoise(BaseNoise):
    """
    Phase flip noise, modeling Z-axis errors on qubits.

    Allows configurable Z (phase flip) and I (identity/no error) probabilities,
    enabling study of phase decoherence and its impact on entanglement.
    """

    def __init__(
        self,
        error_rate: float = DEFAULT_ERROR_RATE,
        num_qubits: int = 1,
        z_prob: float = 0.5,
        i_prob: float = 0.5,
    ):
        """
        Initialize with configurable Z and I probabilities (must sum to 1).

        Args:
            error_rate (float): Base error rate (unused here, but kept for consistency).
            num_qubits (int): Number of qubits (typically 1 for phase flip).
            z_prob (float): Probability of Z (phase flip) error (default 0.5).
            i_prob (float): Probability of I (no error) (default 0.5).
        """
        if not (
            0 <= z_prob <= 1 and 0 <= i_prob <= 1 and abs(z_prob + i_prob - 1) < 1e-10
        ):
            raise ValueError(
                "Z and I probabilities must sum to 1 and be between 0 and 1."
            )
        super().__init__(error_rate, num_qubits)
        self.z_prob = z_prob
        self.i_prob = i_prob

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        noise = pauli_error([("Z", self.z_prob), ("I", self.i_prob)])
        noise_model.add_all_qubit_quantum_error(
            noise, gate_list
        )  # Removed is_inst_supported
        logger.debug(
            f"Applied phase flip noise (Z={self.z_prob}, I={self.i_prob}) to {gate_list}"
        )


class AmplitudeDampingNoise(BaseNoise):
    """
    Amplitude damping noise, modeling energy loss (e.g., qubit relaxation to |0>).

    Useful for studying irreversible energy decay and its effect on quantum states.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        # ✅ Only apply amplitude damping to single-qubit gates
        valid_gates = [
            g for g in gate_list if g in ["id", "u1", "u2", "u3"]
        ]  # Only 1-qubit gates
        if not valid_gates:
            logger.warning(
                f"Skipping Amplitude Damping noise: No valid 1-qubit gates found!"
            )
            return

        noise = amplitude_damping_error(self.error_rate)
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied amplitude damping noise (rate={self.error_rate}) to {valid_gates}"
        )


class PhaseDampingNoise(BaseNoise):
    """
    Phase damping noise, modeling dephasing without energy loss.

    Ideal for studying coherence loss and its geometric constraints in Hilbert space.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        # ✅ Only apply phase damping to single-qubit gates
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]

        if not valid_gates:
            logger.warning(
                f"Skipping Phase Damping noise: No valid 1-qubit gates found!"
            )
            return  # ✅ Avoid error by skipping invalid cases

        noise = phase_damping_error(self.error_rate)
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied phase damping noise (rate={self.error_rate}) to {valid_gates}"
        )


class ThermalRelaxationNoise(BaseNoise):
    """
    Thermal relaxation noise, modeling T1 (energy relaxation) and T2 (dephasing) effects.

    Captures realistic qubit behavior under environmental thermal noise, useful for studying
    entanglement decay and hypergraph correlations.
    """

    def __init__(
        self,
        error_rate: float = DEFAULT_ERROR_RATE,
        num_qubits: int = 1,
        t1: float = DEFAULT_T1,
        t2: float = DEFAULT_T2,
    ):
        """
        Initialize with thermal relaxation times T1 (energy) and T2 (dephasing).

        Args:
            error_rate (float): Base error rate (unused here, but kept for consistency).
            num_qubits (int): Number of qubits affected.
            t1 (float): Relaxation time (seconds, default 100e-6).
            t2 (float): Dephasing time (seconds, default 80e-6).
        """
        super().__init__(error_rate, num_qubits)
        self.t1 = t1
        self.t2 = t2

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        noise = thermal_relaxation_error(
            self.t1, self.t2, 0
        )  # Time=0 for instantaneous effect
        noise_model.add_all_qubit_quantum_error(noise, gate_list)
        logger.debug(
            f"Applied thermal relaxation noise (T1={self.t1}, T2={self.t2}) to {gate_list}"
        )


class BitFlipNoise(BaseNoise):
    """
    Bit flip noise, modeling X-axis errors on qubits.

    Useful for studying random bit flips and their impact on entanglement structure,
    potentially revealing hypergraph patterns.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        valid_gates = [
            g for g in gate_list if g in ["id", "u1", "u2", "u3"]
        ]  # ✅ Only apply to 1-qubit gates
        if not valid_gates:
            logger.warning(f"Skipping Bit Flip noise: No valid 1-qubit gates found!")
            return

        noise = pauli_error([("X", self.error_rate), ("I", 1 - self.error_rate)])
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied bit flip noise (rate={self.error_rate}) to {valid_gates}"
        )


# Noise factory for easy instantiation
NOISE_CLASSES = {
    "DEPOLARIZING": DepolarizingNoise,
    "PHASE_FLIP": PhaseFlipNoise,
    "AMPLITUDE_DAMPING": AmplitudeDampingNoise,
    "PHASE_DAMPING": PhaseDampingNoise,
    "THERMAL_RELAXATION": ThermalRelaxationNoise,
    "BIT_FLIP": BitFlipNoise,
}


def create_noise_model(
    noise_type: str,
    num_qubits: int,
    error_rate: Optional[float] = None,
    z_prob: Optional[float] = None,
    i_prob: Optional[float] = None,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
) -> NoiseModel:
    """
    Creates a scalable, configurable noise model for quantum experiments.

    Supports multiple noise types with customizable parameters, applied to 1+ qubit gates.
    Logs noise effects for hypergraph or fluid dynamics analysis.

    Args:
        noise_type (str): Type of noise (e.g., "DEPOLARIZING", "PHASE_FLIP").
        num_qubits (int): Number of qubits in the experiment.
        error_rate (float, optional): Base error rate (default from config if None).
        z_prob (float, optional): Z probability for PHASE_FLIP (requires i_prob, sums to 1).
        i_prob (float, optional): I probability for PHASE_FLIP (requires z_prob, sums to 1).
        t1 (float, optional): T1 relaxation time for THERMAL_RELAXATION (seconds).
        t2 (float, optional): T2 dephasing time for THERMAL_RELAXATION (seconds).

    Returns:
        NoiseModel: Configured Qiskit noise model.

    Raises:
        ValueError: If noise_type is invalid or parameters are inconsistent.
    """
    if noise_type not in NOISE_CLASSES:
        raise ValueError(
            f"Invalid NOISE_TYPE: {noise_type}. Choose from {list(NOISE_CLASSES.keys())}"
        )

    noise_model = NoiseModel()
    noise_class = NOISE_CLASSES[noise_type]

    # Configure noise parameters based on type
    if noise_type == "PHASE_FLIP":
        if z_prob is None or i_prob is None:
            z_prob = 0.5  # Default Z probability
            i_prob = 0.5  # Default I probability
        noise = noise_class(
            error_rate=DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            z_prob=z_prob,
            i_prob=i_prob,
        )
    elif noise_type == "THERMAL_RELAXATION":
        noise = noise_class(
            error_rate=DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            t1=t1 or DEFAULT_T1,
            t2=t2 or DEFAULT_T2,
        )
    else:
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE, num_qubits=num_qubits
        )

    # Apply noise to appropriate gates, scaling for multi-qubit systems
    for qubits in range(1, num_qubits + 1):
        if qubits == 1:
            gate_list = ["id", "u1", "u2", "u3"]
        elif qubits == 2:
            gate_list = ["cx"]
        else:
            gate_list = ["cx", f"mct_{qubits}"]

        # 🚨 Only apply noise types that match qubit count!
        if noise_type in [
            "DEPOLARIZING",
            "BIT_FLIP",
            "PHASE_FLIP",
            "AMPLITUDE_DAMPING",
            "PHASE_DAMPING",
        ]:
            if qubits == 1:
                noise.apply(noise_model, gate_list)
        elif noise_type == "THERMAL_RELAXATION":  # Thermal relaxation is qubit-wide
            noise.apply(noise_model, gate_list)
        else:
            logger.warning(
                f"Skipping {noise_type} noise for {qubits}-qubit gates (not supported)."
            )

        logger.info(f"Applied {noise_type} noise to {qubits}-qubit gates: {gate_list}")

        # Log correlation impact for hypergraph/fluid analysis
        logger.debug(
            f"Noise effect on correlations: May disrupt {qubits}-qubit entanglement structure, "
            f"potentially preserving parity or symmetry patterns for hypergraph mapping."
        )

    return noise_model


# Example config for batch runs (optional, can be loaded from JSON/YAML)
NOISE_CONFIG = {
    "DEPOLARIZING": {"error_rate": 0.1},
    "PHASE_FLIP": {"z_prob": 0.2, "i_prob": 0.8},
    "AMPLITUDE_DAMPING": {"error_rate": 0.05},
    "PHASE_DAMPING": {"error_rate": 0.05},
    "THERMAL_RELAXATION": {"t1": 100e-6, "t2": 80e-6},
    "BIT_FLIP": {"error_rate": 0.1},
}
