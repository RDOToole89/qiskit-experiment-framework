# src/quantum_experiment/noise_models/depolarizing.py

import logging
from qiskit_aer.noise import depolarizing_error, NoiseModel
from .base_noise import BaseNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class DepolarizingNoise(BaseNoise):
    """
    Depolarizing noise, modeling random bit flips across qubits.

    Scales error rate with qubit count, suitable for studying random decoherence patterns.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        """
        Applies depolarizing noise correctly based on gate size.
        """
        for gate in gate_list:
            if gate in ["id", "u1", "u2", "u3"]:  # 1-qubit gates
                noise = depolarizing_error(self.error_rate, 1)
                noise_model.add_all_qubit_quantum_error(noise, [gate])
            elif gate == "cx":  # 2-qubit gates
                noise = depolarizing_error(self.error_rate, 2)
                noise_model.add_all_qubit_quantum_error(noise, [gate])
            elif "mct" in gate:  # Multi-controlled Toffoli (3+ qubits)
                try:
                    qubit_count = int(gate.split("_")[1])  # e.g., "mct_3"
                    noise = depolarizing_error(self.error_rate, qubit_count)
                    noise_model.add_all_qubit_quantum_error(noise, [gate])
                except (IndexError, ValueError):
                    logger.warning(f"Skipping unknown gate format: {gate}")
                    continue

        logger.debug(f"Applied depolarizing noise (rate={self.error_rate}) to {gate_list}")
