# src/quantum_experiment/state_preparation/w_state.py

from qiskit import QuantumCircuit
import numpy as np
from .base_state import BaseState
import logging

logger = logging.getLogger("QuantumExperiment.StatePreparation")

class WState(BaseState):
    """
    W state preparation (|100…⟩ + |010…⟩ + …)/√N, modeling distributed entanglement.
    Requires at least 2 qubits.
    """

    def create(self) -> QuantumCircuit:
        if self.num_qubits < 2:
            raise ValueError("W state requires at least 2 qubits for meaningful entanglement.")
        qc = QuantumCircuit(self.num_qubits)
        w_state = np.zeros(2**self.num_qubits, dtype=complex)
        for i in range(self.num_qubits):
            w_state[1 << i] = 1 / np.sqrt(self.num_qubits)
        qc.initialize(w_state, range(self.num_qubits))
        logger.debug(f"Created {self.num_qubits}-qubit W state.")
        return qc
