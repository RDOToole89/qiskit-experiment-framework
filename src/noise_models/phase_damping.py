from .base_noise import BaseNoise
from qiskit_aer.noise import phase_damping_error
from qiskit_aer.noise import NoiseModel
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class PhaseDampingNoise(BaseNoise):
    """
    Phase damping noise, modeling dephasing without energy loss.

    Ideal for studying coherence loss and its geometric constraints in Hilbert space.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        # ✅ Only apply phase damping to single-qubit gates
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]

        if not valid_gates:
            logger.warning(
                f"Skipping Phase Damping noise: No valid 1-qubit gates found!"
            )
            return  # ✅ Avoid error by skipping invalid cases

        noise = phase_damping_error(self.error_rate)
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        logger.debug(
            f"Applied phase damping noise (rate={self.error_rate}) to {valid_gates}"
        )
