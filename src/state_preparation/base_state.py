# src/quantum_experiment/state_preparation/base_state.py

from qiskit import QuantumCircuit
from typing import Optional, Dict

class BaseState:
    def __init__(self, num_qubits: int, custom_params: Optional[Dict] = None):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        self.num_qubits = num_qubits
        self.custom_params = custom_params or {}

    def create(self, add_barrier: bool = False) -> QuantumCircuit:
        """
        Creates the quantum circuit for the state.

        Args:
            add_barrier (bool): Whether to add a barrier after state preparation.

        Returns:
            QuantumCircuit: Prepared quantum circuit.

        Raises:
            NotImplementedError: If subclass doesn't implement create.
        """
        raise NotImplementedError("Subclasses must implement create()")
