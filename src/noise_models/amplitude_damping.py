# src/quantum_experiment/noise_models/amplitude_damping.py

import logging
from qiskit_aer.noise import amplitude_damping_error, NoiseModel
from .base_noise import BaseNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class AmplitudeDampingNoise(BaseNoise):
    """
    Amplitude damping noise, modeling energy loss (e.g., qubit relaxation to |0>).

    Useful for studying irreversible energy decay and its effect on quantum states.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        # Only apply amplitude damping to single-qubit gates
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]
        if not valid_gates:
            logger.warning("Skipping Amplitude Damping noise: No valid 1-qubit gates found!")
            return

        noise = amplitude_damping_error(self.error_rate)
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied amplitude damping noise (rate={self.error_rate}) to {valid_gates}"
        )
