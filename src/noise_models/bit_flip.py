# src/quantum_experiment/noise_models/bit_flip.py

from qiskit_aer.noise import pauli_error, NoiseModel
from .base_noise import BaseNoise

class BitFlipNoise(BaseNoise):
    """
    Bit flip noise, modeling X-axis errors on qubits.
    """

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]
        if not valid_gates:
            self.log_noise_application(
                noise_type="BIT_FLIP",
                gates=gate_list,
                extra_info={"warning": "No valid 1-qubit gates found, skipping"}
            )
            return

        noise = pauli_error([("X", self.error_rate), ("I", 1 - self.error_rate)])
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        self.log_noise_application(
            noise_type="BIT_FLIP",
            gates=valid_gates,
            extra_info={"error_rate": self.error_rate}
        )
