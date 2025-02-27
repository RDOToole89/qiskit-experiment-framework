# src/quantum_experiment/noise_models/depolarizing.py

from qiskit_aer.noise import NoiseModel, depolarizing_error
from .base_noise import BaseNoise

class DepolarizingNoise(BaseNoise):
    """
    Depolarizing noise model, applying a depolarizing error to specified gates.
    """

    def apply(self, noise_model: NoiseModel, gates: list) -> None:
        error = depolarizing_error(self.error_rate, self.num_qubits)
        for gate in gates:
            noise_model.add_all_qubit_quantum_error(error, gate)
        # Use the base class logging method
        self.log_noise_application(
            noise_type="DEPOLARIZING",
            gates=gates,
            extra_info={"error_rate": self.error_rate}
        )
