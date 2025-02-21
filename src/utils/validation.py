# src/quantum_experiment/utils/validation.py

import numpy as np
from src.noise_models import NOISE_CLASSES
from src.state_preparation import STATE_CLASSES
import logging

logger = logging.getLogger("QuantumExperiment.Validation")


def validate_inputs(
    num_qubits: int,
    state_type: str,
    noise_type: str,
    sim_mode: str,
    angle: float = None,
    error_rate: float = None,
    z_prob: float = None,
    i_prob: float = None,
    t1: float = None,
    t2: float = None,
) -> None:
    """
    Validates input parameters to ensure correctness.
    """

    # 🔹 Validate number of qubits
    if num_qubits < 1:
        raise ValueError("Number of qubits must be at least 1.")

    # 🔹 Validate state type
    if state_type not in STATE_CLASSES:
        raise ValueError(
            f"Invalid state type: {state_type}. Choose from {list(STATE_CLASSES.keys())}"
        )

    # 🔹 Validate noise type
    if noise_type not in NOISE_CLASSES:
        raise ValueError(
            f"Invalid noise type: {noise_type}. Choose from {list(NOISE_CLASSES.keys())}"
        )

    # 🔹 Validate simulation mode
    if sim_mode not in ["qasm", "density"]:
        raise ValueError(
            f"Invalid simulation mode: {sim_mode}. Choose from ['qasm', 'density']"
        )

    # 🔹 Validate CLUSTER angle (only needed for CLUSTER state)
    if state_type == "CLUSTER" and angle is not None:
        if not (0 <= angle <= 2 * np.pi):
            raise ValueError(
                "Angle for CLUSTER state must be between 0 and 2π radians."
            )

    # 🔹 Validate noise parameters
    if error_rate is not None and not (0 <= error_rate <= 1):
        raise ValueError("Error rate must be between 0 and 1.")

    # 🔹 Validate PHASE_FLIP noise parameters (z_prob and i_prob must sum to 1)
    if noise_type == "PHASE_FLIP" and (z_prob is not None or i_prob is not None):
        if z_prob is None or i_prob is None:
            raise ValueError(
                "Both z_prob and i_prob must be provided for PHASE_FLIP noise."
            )
        if not (
            0 <= z_prob <= 1 and 0 <= i_prob <= 1 and abs(z_prob + i_prob - 1) < 1e-10
        ):
            raise ValueError(
                "Z and I probabilities for PHASE_FLIP must sum to 1 and be between 0 and 1."
            )

    # 🔹 Validate THERMAL_RELAXATION noise parameters
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
        f"✅ Validated inputs: num_qubits={num_qubits}, state_type={state_type}, "
        f"noise_type={noise_type}, sim_mode={sim_mode}, angle={angle}, error_rate={error_rate}, "
        f"z_prob={z_prob}, i_prob={i_prob}, t1={t1}, t2={t2}"
    )
