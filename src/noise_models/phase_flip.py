from .base_noise import BaseNoise
from qiskit_aer.noise import pauli_error


class PhaseFlipNoise(BaseNoise):
    """Applies phase flip noise with configurable Z and I probabilities."""

    def __init__(self, error_rate=0.1, num_qubits=1, z_prob=0.5, i_prob=0.5):
        super().__init__(error_rate, num_qubits)
        if not (
            0 <= z_prob <= 1 and 0 <= i_prob <= 1 and abs(z_prob + i_prob - 1) < 1e-10
        ):
            raise ValueError("Z and I probabilities must sum to 1.")
        self.z_prob = z_prob
        self.i_prob = i_prob

    def apply(self, noise_model, gate_list):
        noise = pauli_error([("Z", self.z_prob), ("I", self.i_prob)])
        noise_model.add_all_qubit_quantum_error(noise, gate_list)
