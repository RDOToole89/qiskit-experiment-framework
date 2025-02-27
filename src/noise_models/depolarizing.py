# src/noise_models/depolarizing.py

from qiskit_aer.noise import NoiseModel, depolarizing_error
from .base_noise import BaseNoise

class DepolarizingNoise(BaseNoise):
    """
    Depolarizing noise model, applying a depolarizing error to specified gates.
    """
    def apply(self, noise_model: NoiseModel, gates: list, qubits_for_error: int = None) -> None:
        # Use qubits_for_error if provided, otherwise use self.num_qubits
        num_qubits_for_error = qubits_for_error if qubits_for_error is not None else self.num_qubits

        # Create a depolarizing error for the appropriate number of qubits
        error = depolarizing_error(self.error_rate, num_qubits_for_error)
        for gate in gates:
            noise_model.add_all_qubit_quantum_error(error, gate)
        self.log_noise_application(
            noise_type="DEPOLARIZING",
            gates=gates,
            extra_info={"error_rate": self.error_rate}
        )
