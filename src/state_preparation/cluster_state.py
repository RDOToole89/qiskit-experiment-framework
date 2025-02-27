# src/quantum_experiment/state_preparation/cluster_state.py

from qiskit import QuantumCircuit
import numpy as np
from .base_state import BaseState
import logging

logger = logging.getLogger("QuantumExperiment.StatePreparation")

class ClusterState(BaseState):
    """
    Cluster state preparation, modeling multipartite entanglement on a lattice.
    Supports 1D or 2D lattices via custom_params["lattice"].
    """

    def create(self) -> QuantumCircuit:
        if self.num_qubits < 2:
            raise ValueError("Cluster state requires at least 2 qubits for meaningful entanglement.")
        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))
        # Default to 1D lattice
        if "lattice" in self.custom_params:
            lattice = self.custom_params["lattice"].lower()
            if lattice == "2d":
                if self.num_qubits % 2 != 0:
                    raise ValueError("2D lattice requires an even number of qubits for a square grid.")
                rows = cols = int(np.sqrt(self.num_qubits))
                for i in range(rows):
                    for j in range(cols):
                        idx = i * cols + j
                        if j < cols - 1:
                            qc.cz(idx, idx + 1)
                        if i < rows - 1:
                            qc.cz(idx, idx + cols)
            else:  # Assume 1D or default
                for i in range(self.num_qubits - 1):
                    qc.cz(i, i + 1)
        else:
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)
        logger.debug(
            f"Created {self.num_qubits}-qubit Cluster state (lattice: {'2d' if 'lattice' in self.custom_params and self.custom_params['lattice'].lower() == '2d' else '1d'})"
        )
        return qc
