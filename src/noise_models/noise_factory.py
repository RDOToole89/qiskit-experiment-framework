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
    simulate_density: bool = False,
) -> NoiseModel:
    """
    Creates a scalable, configurable noise model for quantum experiments.
    
    When simulate_density is True (e.g. for density matrix simulation),
    only multi-qubit (2+ qubit) errors are added—this mimics the old behavior that
    worked in density mode (which only added a "cx" error).
    """
    if noise_type not in NOISE_CLASSES:
        raise ValueError(
            f"Invalid NOISE_TYPE: {noise_type}. Choose from {list(NOISE_CLASSES.keys())}"
        )

    noise_model = NoiseModel()
    noise_class = NOISE_CLASSES[noise_type]

    # Instantiate the noise object with parameters specific to the noise type.
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

    # In density mode, avoid applying one-qubit errors which might get composed
    # with decompositions of multi-qubit gates. (The old working version for density
    # mode only applied a noise channel for the "cx" instruction.)
    for qubits in range(1, num_qubits + 1):
        if simulate_density:
            if qubits == 2:
                gate_list = ["cx"]
            elif qubits > 2:
                gate_list = [f"mct_{qubits}"]
            else:
                # Skip one-qubit noise in density simulation mode.
                continue
        else:
            if qubits == 1:
                gate_list = ["id", "u1", "u2", "u3"]
            elif qubits == 2:
                gate_list = ["cx"]
            else:
                gate_list = [f"mct_{qubits}"]

        try:
            noise.apply(noise_model, gate_list)
            logger.info(f"Applied {noise_type} noise to {qubits}-qubit gates: {gate_list}")
        except Exception as e:
            logger.warning(
                f"Failed to apply {noise_type} noise for {qubits}-qubit gates: {gate_list}. Error: {e}"
            )

    return noise_model
