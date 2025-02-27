# src/quantum_experiment/noise_models/phase_flip.py

from qiskit_aer.noise import NoiseModel, pauli_error
from .base_noise import BaseNoise

class PhaseFlipNoise(BaseNoise):
    """
    Phase flip noise model, applying a phase flip error to specified gates.
    """

    def __init__(self, error_rate: float, num_qubits: int, z_prob: float, i_prob: float, experiment_id: str = "N/A"):
        super().__init__(error_rate=error_rate, num_qubits=num_qubits, experiment_id=experiment_id)
        self.z_prob = z_prob
        self.i_prob = i_prob

    def apply(self, noise_model: NoiseModel, gates: list) -> None:
        error = pauli_error([("Z", self.z_prob), ("I", self.i_prob)])
        for gate in gates:
            noise_model.add_all_qubit_quantum_error(error, gate)
        self.log_noise_application(
            noise_type="PHASE_FLIP",
            gates=gates,
            extra_info={"z_prob": self.z_prob, "i_prob": self.i_prob}
        )
