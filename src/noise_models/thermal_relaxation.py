# src/noise_models/thermal_relaxation.py

from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from .base_noise import BaseNoise

class ThermalRelaxationNoise(BaseNoise):
    """
    Thermal relaxation noise model, applying T1/T2 relaxation errors to specified gates.
    """
    def __init__(self, error_rate: float, num_qubits: int, t1: float, t2: float, experiment_id: str = "N/A"):
        super().__init__(error_rate=error_rate, num_qubits=num_qubits, experiment_id=experiment_id)
        self.t1 = t1
        self.t2 = t2

    def apply(self, noise_model: NoiseModel, gates: list, qubits_for_error: int = None) -> None:
        # Thermal relaxation noise is always 1-qubit, so ignore qubits_for_error
        error = thermal_relaxation_error(self.t1, self.t2, self.error_rate)
        for gate in gates:
            noise_model.add_all_qubit_quantum_error(error, gate)
        self.log_noise_application(
            noise_type="THERMAL_RELAXATION",
            gates=gates,
            extra_info={"t1": self.t1, "t2": self.t2}
        )
