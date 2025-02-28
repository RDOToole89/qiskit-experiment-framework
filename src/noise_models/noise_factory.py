# src/noise_models/noise_factory.py

import logging
from typing import Optional, List
from qiskit_aer.noise import NoiseModel
from src.utils import logger as logger_utils
from .base_noise import BaseNoise
from .depolarizing import DepolarizingNoise
from .phase_flip import PhaseFlipNoise
from .amplitude_damping import AmplitudeDampingNoise
from .phase_damping import PhaseDampingNoise
from .thermal_relaxation import ThermalRelaxationNoise
from .bit_flip import BitFlipNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")

# Example defaults (you might want to move these to config if needed)
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
    "PHASE_FLIP": {"error_rate": 0.1},
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
    experiment_id: str = "N/A",
) -> NoiseModel:
    """
    Creates a scalable, configurable noise model for quantum experiments.

    When simulate_density is True (e.g. for density matrix simulation),
    only multi-qubit (2+ qubit) errors are added—this mimics the old behavior that
    worked in density mode (which only added a "cx" error).

    Args:
        noise_type (str): Type of noise to apply.
        num_qubits (int): Number of qubits in the circuit.
        error_rate (float, optional): Custom error rate for noise models.
        z_prob (float, optional): Z probability for custom noise models (e.g., DEPOLARIZING).
        i_prob (float, optional): I probability for custom noise models (e.g., DEPOLARIZING).
        t1 (float, optional): T1 relaxation time for THERMAL_RELAXATION noise.
        t2 (float, optional): T2 dephasing time for THERMAL_RELAXATION noise.
        simulate_density (bool): Whether to simulate density matrix mode.
        experiment_id (str): Unique identifier for this experiment run.

    Returns:
        NoiseModel: Configured noise model.

    Raises:
        ValueError: If the noise type or parameters are invalid.
    """
    if noise_type not in NOISE_CLASSES:
        raise ValueError(
            f"Invalid NOISE_TYPE: {noise_type}. Choose from {list(NOISE_CLASSES.keys())}"
        )

    noise_model = NoiseModel()
    noise_class = NOISE_CLASSES[noise_type]

    # Instantiate the noise object with parameters specific to the noise type
    if noise_type == "THERMAL_RELAXATION":
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            t1=t1 or DEFAULT_T1,
            t2=t2 or DEFAULT_T2,
            experiment_id=experiment_id,
        )
    elif noise_type == "PHASE_FLIP":
        # Explicitly pass z_prob and i_prob as None if not provided
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            z_prob=z_prob,
            i_prob=i_prob,
            experiment_id=experiment_id,
        )
    else:
        # For other noise types (DEPOLARIZING, AMPLITUDE_DAMPING, etc.), z_prob and i_prob are not used
        noise = noise_class(
            error_rate=error_rate or DEFAULT_ERROR_RATE,
            num_qubits=num_qubits,
            experiment_id=experiment_id,
        )

    # Define gate lists and their corresponding qubit counts
    gate_configs = []
    single_qubit_noise_types = [
        "PHASE_FLIP",
        "AMPLITUDE_DAMPING",
        "PHASE_DAMPING",
        "BIT_FLIP",
    ]

    if noise_type in single_qubit_noise_types:
        # Apply single-qubit noise to each qubit individually
        for qubit in range(num_qubits):
            gate_configs.append(
                {"qubits": 1, "gates": ["id"], "target_qubits": [qubit]}
            )
    else:
        # For multi-qubit noise types like DEPOLARIZING and THERMAL_RELAXATION
        if simulate_density:
            if num_qubits >= 2:
                gate_configs.append({"qubits": 2, "gates": ["cx"]})
            if num_qubits > 2:
                gate_configs.append(
                    {"qubits": num_qubits, "gates": [f"mct_{num_qubits}"]}
                )
        else:
            gate_configs.extend(
                [
                    {"qubits": 1, "gates": ["id", "u1", "u2", "u3"]},
                    {"qubits": 2, "gates": ["cx"]},
                    {"qubits": num_qubits, "gates": [f"mct_{num_qubits}"]},
                ]
            )

    # Apply noise to gates
    for config in gate_configs:
        qubits = config["qubits"]
        gate_list = config["gates"]
        target_qubits = config.get("target_qubits", None)

        # Skip if the noise type is single-qubit but the gate requires more qubits (for non-single-qubit noise types)
        if noise_type in single_qubit_noise_types and qubits > 1:
            logger_utils.log_with_experiment_id(
                logger,
                "info",
                (
                    f"Skipping {noise_type} noise for {qubits}-qubit gates {gate_list}: "
                    "This noise type only supports single-qubit gates. "
                    "Use a multi-qubit noise type like DEPOLARIZING for these gates."
                ),
                experiment_id,
                extra_info={
                    "noise_type": noise_type,
                    "qubits": qubits,
                    "gates": gate_list,
                },
            )
            continue

        try:
            if target_qubits is not None:
                # Apply noise to specific qubits (for single-qubit noise)
                noise.apply(
                    noise_model,
                    gate_list,
                    qubits_for_error=qubits,
                    specific_qubits=target_qubits,
                )
                logger_utils.log_with_experiment_id(
                    logger,
                    "info",
                    f"Applied {noise_type} noise to qubit {target_qubits} on gates: {gate_list}",
                    experiment_id,
                    extra_info={
                        "noise_type": noise_type,
                        "qubits": target_qubits,
                        "gates": gate_list,
                    },
                )
            else:
                # Apply noise to gates with specified qubit count
                noise.apply(noise_model, gate_list, qubits_for_error=qubits)
                logger_utils.log_with_experiment_id(
                    logger,
                    "info",
                    f"Applied {noise_type} noise to {qubits}-qubit gates: {gate_list}",
                    experiment_id,
                    extra_info={
                        "noise_type": noise_type,
                        "qubits": qubits,
                        "gates": gate_list,
                    },
                )
        except Exception as e:
            logger_utils.log_with_experiment_id(
                logger,
                "warning",
                (
                    f"Failed to apply {noise_type} noise to {qubits}-qubit gates {gate_list}. "
                    f"Error: {str(e)}. This may be due to an incompatible qubit count. "
                    "Ensure the noise type matches the gate's qubit requirements."
                ),
                experiment_id,
                extra_info={
                    "noise_type": noise_type,
                    "qubits": qubits,
                    "gates": gate_list,
                    "error": str(e),
                },
            )

    return noise_model
