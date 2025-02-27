# src/noise_models/phase_damping.py

from qiskit_aer.noise import phase_damping_error, NoiseModel
from .base_noise import BaseNoise

class PhaseDampingNoise(BaseNoise):
    """
    Phase damping noise, modeling dephasing without energy loss.
    Ideal for studying coherence loss in Hilbert space.
    """
    def apply(self, noise_model: NoiseModel, gate_list: list, qubits_for_error: int = None) -> None:
        valid_gates = [g for g in gate_list if g in ["id", "u1", "u2", "u3"]]
        if not valid_gates:
            self.log_noise_application(
                noise_type="PHASE_DAMPING",
                gates=gate_list,
                extra_info={"warning": "No valid 1-qubit gates found, skipping"}
            )
            return

        # Phase damping is always 1-qubit, so ignore qubits_for_error
        noise = phase_damping_error(self.error_rate)
        noise_model.add_all_qubit_quantum_error(noise, valid_gates)
        self.log_noise_application(
            noise_type="PHASE_DAMPING",
            gates=valid_gates,
            extra_info={"error_rate": self.error_rate}
        )
