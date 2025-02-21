from .base_noise import BaseNoise
from qiskit_aer.noise import pauli_error
from qiskit_aer.noise import NoiseModel
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")


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
