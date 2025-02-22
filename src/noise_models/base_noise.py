from qiskit_aer.noise import NoiseModel
from src.config import DEFAULT_ERROR_RATE
import logging

logger = logging.getLogger("QuantumExperiment.NoiseModels")


class BaseNoise:
    """
    Base class for all noise models, providing a template for noise application.

    Attributes:
        error_rate (float): Probability of error occurrence (configurable).
        num_qubits (int): Number of qubits affected by the noise.
    """

    def __init__(self, error_rate: float = DEFAULT_ERROR_RATE, num_qubits: int = 1):
        self.error_rate = error_rate
        self.num_qubits = num_qubits

    def apply(self, noise_model: NoiseModel, gate_list: list) -> None:
        """
        Applies noise to the specified gates in the noise model.

        Args:
            noise_model (NoiseModel): Qiskit noise model to modify.
            gate_list (list): List of gate names (e.g., ['id', 'u1']) to apply noise to.
        """
        raise NotImplementedError("Subclasses must implement apply()")
