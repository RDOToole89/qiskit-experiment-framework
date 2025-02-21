# src/state_preparation/cluster_state.py

from qiskit import QuantumCircuit
import numpy as np
import logging
from .base_state import BaseState

logger = logging.getLogger("QuantumExperiment.ClusterState")


class ClusterState(BaseState):
    """
    Cluster state preparation, modeling multipartite entanglement on a lattice.

    Supports 1D or 2D lattices via custom_params["lattice"].
    """

    def create(self) -> QuantumCircuit:
        if self.num_qubits < 2:
            raise ValueError("Cluster state requires at least 2 qubits.")

        qc = QuantumCircuit(self.num_qubits)
        qc.h(range(self.num_qubits))

        lattice = self.custom_params.get("lattice", "1d").lower()
        if lattice == "2d":
            if self.num_qubits % 2 != 0:
                raise ValueError("2D lattice requires an even number of qubits.")
            rows = cols = int(np.sqrt(self.num_qubits))
            for i in range(rows):
                for j in range(cols):
                    idx = i * cols + j
                    if j < cols - 1:
                        qc.cz(idx, idx + 1)
                    if i < rows - 1:
                        qc.cz(idx, idx + cols)
        else:
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)

        logger.debug(
            f"Created {self.num_qubits}-qubit cluster state (lattice: {lattice})."
        )
        return qc
