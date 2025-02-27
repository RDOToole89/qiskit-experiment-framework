# src/state_preparation/base_state.py

from qiskit import QuantumCircuit
from typing import Optional, Dict
import logging

logger = logging.getLogger("QuantumExperiment.StatePreparation")

class BaseState:
    """
    Base class for quantum state preparation, providing a template for state creation.

    Attributes:
        num_qubits (int): Number of qubits in the state.
        custom_params (dict, optional): Custom parameters for state customization.
        experiment_id (str): Unique identifier for the experiment run.
    """

    def __init__(self, num_qubits: int, custom_params: Optional[Dict] = None, experiment_id: str = "N/A"):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        self.num_qubits = num_qubits
        self.custom_params = custom_params or {}
        self.experiment_id = experiment_id

    def create(self, add_barrier: bool = False, experiment_id: str = "N/A") -> QuantumCircuit:
        """
        Creates the quantum circuit for the state.

        Args:
            add_barrier (bool): Whether to add a barrier after state preparation.
            experiment_id (str): Unique identifier for the experiment run.

        Returns:
            QuantumCircuit: Prepared quantum circuit.

        Raises:
            NotImplementedError: If subclass doesn't implement create.
        """
        raise NotImplementedError("Subclasses must implement create()")

    def log_state_creation(self, state_type: str, extra_info: Optional[dict] = None) -> None:
        """
        Logs the creation of the quantum state using structured logging.

        Args:
            state_type (str): Type of state being created (e.g., "GHZ").
            extra_info (dict, optional): Additional metadata to include in the log.
        """
        from src.utils.logger import log_with_experiment_id
        base_info = {"state_type": state_type, "num_qubits": self.num_qubits}
        if extra_info:
            base_info.update(extra_info)
        log_with_experiment_id(
            logger, "debug",
            f"Created {self.num_qubits}-qubit {state_type} state.",
            self.experiment_id,
            extra_info=base_info
        )
