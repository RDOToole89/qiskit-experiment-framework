# src/quantum_experiment/noise_models/bit_flip.py

import logging
from qiskit_aer.noise import pauli_error, NoiseModel
from .base_noise import BaseNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class BitFlipNoise(BaseNoise):
    """
    Bit flip noise, modeling X-axis errors on qubits.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]
        if not valid_gates:
            logger.warning("Skipping Bit Flip noise: No valid 1-qubit gates found!")
            return

        noise = pauli_error([("X", self.error_rate), ("I", 1 - self.error_rate)])
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied bit flip noise (rate={self.error_rate}) to {valid_gates}"
        )
