# src/quantum_experiment/noise_models/thermal_relaxation.py

import logging
from qiskit_aer.noise import thermal_relaxation_error, NoiseModel
from .base_noise import BaseNoise

logger = logging.getLogger("QuantumExperiment.NoiseModels")

DEFAULT_T1 = 100e-6
DEFAULT_T2 = 80e-6


class ThermalRelaxationNoise(BaseNoise):
    """
    Thermal relaxation noise, modeling T1 (energy relaxation) and T2 (dephasing) effects.
    """

    def __init__(
        self,
        error_rate: float = 0.1,
        num_qubits: int = 1,
        t1: float = DEFAULT_T1,
        t2: float = DEFAULT_T2,
    ):
        super().__init__(error_rate, num_qubits)
        self.t1 = t1
        self.t2 = t2

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        noise = thermal_relaxation_error(self.t1, self.t2, 0)
        noise_model.add_all_qubit_quantum_error(noise, gate_list)
        logger.debug(
            f"Applied thermal relaxation noise (T1={self.t1}, T2={self.t2}) to {gate_list}"
        )
