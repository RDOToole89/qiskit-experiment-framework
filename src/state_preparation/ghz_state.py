# src/state_preparation/ghz_state.py

from qiskit import QuantumCircuit
from .base_state import BaseState

class GHZState(BaseState):
    """
    GHZ state preparation (|000…⟩ + |111…⟩)/√2, modeling multipartite entanglement.
    """

    def create(self, add_barrier: bool = False, experiment_id: str = "N/A") -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.h(0)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        if add_barrier:
            qc.barrier()
        self.log_state_creation(state_type="GHZ")
        return qc
