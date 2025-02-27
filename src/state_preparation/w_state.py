# src/state_preparation/w_state.py

from qiskit import QuantumCircuit
import numpy as np
from .base_state import BaseState

class WState(BaseState):
    """
    W state preparation (e.g., (|100⟩ + |010⟩ + |001⟩)/√3 for 3 qubits).
    """

    def create(self, add_barrier: bool = False, experiment_id: str = "N/A") -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Initialize W-state
        w_state = np.zeros(2**self.num_qubits, dtype=complex)
        for i in range(self.num_qubits):
            w_state[1 << i] = 1 / np.sqrt(self.num_qubits)
        qc.initialize(w_state, range(self.num_qubits))
        if add_barrier:
            qc.barrier()
        self.log_state_creation(state_type="W")
        return qc
