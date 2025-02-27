# src/state_preparation/cluster_state.py

from qiskit import QuantumCircuit
from .base_state import BaseState

class ClusterState(BaseState):
    """
    Cluster state preparation for a 1D or 2D lattice.
    """

    def create(self, add_barrier: bool = False, experiment_id: str = "N/A") -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Basic 1D cluster state preparation
        lattice = self.custom_params.get("lattice", "1d")
        qc.h(range(self.num_qubits))
        for i in range(self.num_qubits - 1):
            qc.cz(i, i + 1)
        if lattice == "2d":
            # Add additional CZ gates for 2D lattice (simplified example)
            for i in range(self.num_qubits - 2):
                qc.cz(i, i + 2)
        if add_barrier:
            qc.barrier()
        self.log_state_creation(state_type="CLUSTER", extra_info={"lattice": lattice})
        return qc
