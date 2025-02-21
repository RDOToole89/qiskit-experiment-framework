from .base_noise import BaseNoise
from qiskit_aer.noise import depolarizing_error
from qiskit_aer.noise import NoiseModel
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")


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
