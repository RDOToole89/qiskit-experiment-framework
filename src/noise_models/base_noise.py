# src/noise_models/base_noise.py

from qiskit_aer.noise import NoiseModel
import logging
from src.utils import logger as logger_utils
from typing import Optional

logger = logging.getLogger("QuantumExperiment.NoiseModels")

# Default error rate for base noise (could also import from a config file)
DEFAULT_ERROR_RATE = 0.1

class BaseNoise:
    """
    Base class for all noise models, providing a template for noise application.

    Attributes:
        error_rate (float): Probability of error occurrence (configurable).
        num_qubits (int): Number of qubits affected by the noise.
        experiment_id (str): Unique identifier for the experiment run.
    """
    def __init__(self, error_rate: float = DEFAULT_ERROR_RATE, num_qubits: int = 1, experiment_id: str = "N/A"):
        self.error_rate = error_rate
        self.num_qubits = num_qubits
        self.experiment_id = experiment_id

    def apply(self, noise_model: NoiseModel, gate_list: list, qubits_for_error: int = None) -> None:
        """
        Applies noise to the specified gates in the noise model.

        Args:
            noise_model (NoiseModel): Qiskit noise model to modify.
            gate_list (list): List of gate names (e.g., ['id', 'u1']) to apply noise to.
            qubits_for_error (int, optional): Number of qubits for the noise error.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def log_noise_application(self, noise_type: str, gates: list, extra_info: Optional[dict] = None) -> None:
        """
        Logs the application of noise to gates using structured logging.

        Args:
            noise_type (str): Type of noise being applied (e.g., "DEPOLARIZING").
            gates (list): List of gates to which noise was applied.
            extra_info (dict, optional): Additional metadata to include in the log.
        """
        base_info = {"noise_type": noise_type, "gates": gates}
        if extra_info:
            base_info.update(extra_info)
        logger_utils.log_with_experiment_id(
            logger, "debug",
            f"Applied {noise_type} noise to {gates}",
            self.experiment_id,
            extra_info=base_info
        )
