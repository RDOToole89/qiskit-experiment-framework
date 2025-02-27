# src/quantum_experiment/state_preparation/ghz_state.py

from qiskit import QuantumCircuit
from .base_state import BaseState
import logging

logger = logging.getLogger("QuantumExperiment.StatePreparation")

class GHZState(BaseState):
    """
    GHZ state preparation (|000…⟩ + |111…⟩)/√2, modeling multipartite entanglement.
    """

    def create(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.h(0)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        logger.debug(f"Created {self.num_qubits}-qubit GHZ state.")
        return qc
