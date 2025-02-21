from .base_noise import BaseNoise
from qiskit_aer.noise import thermal_relaxation_error
from qiskit_aer.noise import NoiseModel
from src.config import DEFAULT_ERROR_RATE, DEFAULT_T1, DEFAULT_T2
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class ThermalRelaxationNoise(BaseNoise):
    """
    Thermal relaxation noise, modeling T1 (energy relaxation) and T2 (dephasing) effects.

    Captures realistic qubit behavior under environmental thermal noise, useful for studying
    entanglement decay and hypergraph correlations.
    """

    def __init__(
        self,
        error_rate: float = DEFAULT_ERROR_RATE,
        num_qubits: int = 1,
        t1: float = DEFAULT_T1,
        t2: float = DEFAULT_T2,
    ):
        """
        Initialize with thermal relaxation times T1 (energy) and T2 (dephasing).

        Args:
            error_rate (float): Base error rate (unused here, but kept for consistency).
            num_qubits (int): Number of qubits affected.
            t1 (float): Relaxation time (seconds, default 100e-6).
            t2 (float): Dephasing time (seconds, default 80e-6).
        """
        super().__init__(error_rate, num_qubits)
        self.t1 = t1
        self.t2 = t2

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        noise = thermal_relaxation_error(
            self.t1, self.t2, 0
        )  # Time=0 for instantaneous effect
        noise_model.add_all_qubit_quantum_error(noise, gate_list)
        logger.debug(
            f"Applied thermal relaxation noise (T1={self.t1}, T2={self.t2}) to {gate_list}"
        )
