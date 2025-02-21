from qiskit_aer.noise import NoiseModel
from .depolarizing import DepolarizingNoise
from .phase_flip import PhaseFlipNoise
from .amplitude_damping import AmplitudeDampingNoise
from .phase_damping import PhaseDampingNoise
from .thermal_relaxation import ThermalRelaxationNoise
from .bit_flip import BitFlipNoise
from src.config import DEFAULT_ERROR_RATE, DEFAULT_T1, DEFAULT_T2
from typing import Optional
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")

# Noise factory for easy instantiation
NOISE_CLASSES = {
    "DEPOLARIZING": DepolarizingNoise,
    "PHASE_FLIP": PhaseFlipNoise,
    "AMPLITUDE_DAMPING": AmplitudeDampingNoise,
    "PHASE_DAMPING": PhaseDampingNoise,
    "THERMAL_RELAXATION": ThermalRelaxationNoise,
    "BIT_FLIP": BitFlipNoise,
}

# Example config for batch runs (optional, can be loaded from JSON/YAML)
NOISE_CONFIG = {
    "DEPOLARIZING": {"error_rate": 0.1},
    "PHASE_FLIP": {"z_prob": 0.2, "i_prob": 0.8},
    "AMPLITUDE_DAMPING": {"error_rate": 0.05},
    "PHASE_DAMPING": {"error_rate": 0.05},
    "THERMAL_RELAXATION": {"t1": 100e-6, "t2": 80e-6},
    "BIT_FLIP": {"error_rate": 0.1},
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
