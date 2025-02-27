# src/quantum_experiment/noise_models/noise_factory.py

import logging
from typing import Optional
from qiskit_aer.noise import NoiseModel
from .base_noise import BaseNoise
from .depolarizing import DepolarizingNoise
from .phase_flip import PhaseFlipNoise
from .amplitude_damping import AmplitudeDampingNoise
from .phase_damping import PhaseDampingNoise
from .thermal_relaxation import ThermalRelaxationNoise
from .bit_flip import BitFlipNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")

# Example defaults
DEFAULT_ERROR_RATE = 0.1
DEFAULT_T1 = 100e-6
DEFAULT_T2 = 80e-6

NOISE_CLASSES = {
    "DEPOLARIZING": DepolarizingNoise,
    "PHASE_FLIP": PhaseFlipNoise,
    "AMPLITUDE_DAMPING": AmplitudeDampingNoise,
    "PHASE_DAMPING": PhaseDampingNoise,
    "THERMAL_RELAXATION": ThermalRelaxationNoise,
    "BIT_FLIP": BitFlipNoise,
}

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
    """
    if noise_type not in NOISE_CLASSES:
        raise ValueError(
            f"Invalid NOISE_TYPE: {noise_type}. Choose from {list(NOISE_CLASSES.keys())}"
        )

    noise_model = NoiseModel()
    noise_class = NOISE_CLASSES[noise_type]

    # Instantiate the noise
    if noise_type == "PHASE_FLIP":
        if z_prob is None or i_prob is None:
            z_prob = 0.5
            i_prob = 0.5
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            z_prob=z_prob,
            i_prob=i_prob,
        )
    elif noise_type == "THERMAL_RELAXATION":
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            t1=t1 or DEFAULT_T1,
            t2=t2 or DEFAULT_T2,
        )
    else:
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
        )

    # Apply noise to appropriate gates
    for qubits in range(1, num_qubits + 1):
        if qubits == 1:
            gate_list = ["id", "u1", "u2", "u3"]
        elif qubits == 2:
            gate_list = ["cx"]
        else:
            gate_list = ["cx", f"mct_{qubits}"]

        # For certain noise types, only apply to 1-qubit gates
        if noise_type in ["DEPOLARIZING", "BIT_FLIP", "PHASE_FLIP", "AMPLITUDE_DAMPING", "PHASE_DAMPING"]:
            if qubits == 1:
                noise.apply(noise_model, gate_list)
        elif noise_type == "THERMAL_RELAXATION":
            noise.apply(noise_model, gate_list)
        else:
            logger.warning(f"Skipping {noise_type} noise for {qubits}-qubit gates (not supported).")

        logger.info(f"Applied {noise_type} noise to {qubits}-qubit gates: {gate_list}")

    return noise_model
