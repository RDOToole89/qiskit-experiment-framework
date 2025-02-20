#!/usr/bin/env python3
# src/quantum_experiment/state_preparation.py

"""
State preparation functions for quantum experiments, designed for extensibility and research integration.

This module prepares quantum states (GHZ, W, CLUSTER, and future types) for experiments, integrating with
a scalable "quantum experimental lab." Supports custom parameters for research (e.g., hypergraph correlations,
fluid dynamics in Hilbert space) and logging of entanglement structure.

Key features:
- Object-oriented design with state classes for maximum extensibility (e.g., new state types).
- Configurable state parameters (e.g., lattice for CLUSTER) for research flexibility.
- Logging of entanglement patterns for hypergraph mapping or density matrix evolution.
- Scalable for multi-qubit systems, with validation for meaningful states.
- Extensible for custom states, gates, or backends via `custom_params`.

Example usage:
    qc = prepare_state("CLUSTER", num_qubits=3)  # 1D cluster state for hypergraph correlations
    # Or with custom params for research:
    qc = prepare_state("CLUSTER", num_qubits=4, custom_params={"lattice": "2d", "hypergraph_data": {"entanglement": [...]}})
"""

from qiskit import QuantumCircuit
import numpy as np
from typing import Optional, Dict
import logging

# Configure logger for state-specific debugging
logger = logging.getLogger("QuantumExperiment.StatePreparation")


class BaseState:
    """
    Base class for quantum state preparation, providing a template for state creation.

    Attributes:
        num_qubits (int): Number of qubits in the state.
        custom_params (dict, optional): Custom parameters for state customization.
    """

    def __init__(self, num_qubits: int, custom_params: Optional[Dict] = None):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1.")
        self.num_qubits = num_qubits
        self.custom_params = custom_params or {}

    def create(self) -> QuantumCircuit:
        """
        Creates the quantum circuit for the state.

        Returns:
            QuantumCircuit: Prepared quantum circuit.

        Raises:
            NotImplementedError: If subclass doesn't implement `create`.
        """
        raise NotImplementedError("Subclasses must implement create()")


class GHZState(BaseState):
    """
    GHZ state preparation (|000…⟩ + |111…⟩)/√2, modeling multipartite entanglement.

    Useful for studying global correlations and structured decoherence patterns (e.g., parity preservation).
    """

    def create(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.h(0)
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        logger.debug(
            f"Created {self.num_qubits}-qubit GHZ state, linking all qubits for global entanglement"
        )
        return qc


class WState(BaseState):
    """
    W state preparation (|100…⟩ + |010…⟩ + …)/√N, modeling distributed entanglement.

    Requires at least 2 qubits for meaningful superposition, ideal for studying robust decoherence.
    """

    def create(self) -> QuantumCircuit:
        if self.num_qubits < 2:
            raise ValueError(
                "W state requires at least 2 qubits for meaningful entanglement."
            )
        qc = QuantumCircuit(self.num_qubits)
        w_state = np.zeros(2**self.num_qubits, dtype=complex)
        for i in range(self.num_qubits):
            w_state[1 << i] = 1 / np.sqrt(self.num_qubits)
        qc.initialize(w_state, range(self.num_qubits))
        logger.debug(
            f"Created {self.num_qubits}-qubit W state, distributing entanglement across qubits"
        )
        return qc


class ClusterState(BaseState):
    """
    Cluster state preparation, modeling multipartite entanglement on a lattice.

    Useful for studying hypergraph correlations (e.g., qubit connectivity) and fluid dynamics
    in Hilbert space, ideal for measurement-based quantum computing and decoherence analysis.
    Supports 1D or 2D lattices via custom_params["lattice"].
    """

    def create(self) -> QuantumCircuit:
        if self.num_qubits < 2:
            raise ValueError(
                "Cluster state requires at least 2 qubits for meaningful entanglement."
            )
        qc = QuantumCircuit(self.num_qubits)
        # Create Hadamard on all qubits
        qc.h(range(self.num_qubits))
        # Default to 1D lattice (adjacent CZ gates)
        if "lattice" in self.custom_params:
            lattice = self.custom_params["lattice"].lower()
            if lattice == "2d":
                if self.num_qubits % 2 != 0:
                    raise ValueError(
                        "2D lattice requires even number of qubits for a square grid."
                    )
                rows = cols = int(np.sqrt(self.num_qubits))
                for i in range(rows):
                    for j in range(cols):
                        idx = i * cols + j
                        if j < cols - 1:
                            qc.cz(idx, idx + 1)  # Horizontal
                        if i < rows - 1:
                            qc.cz(idx, idx + cols)  # Vertical
            else:  # 1D or default
                for i in range(self.num_qubits - 1):
                    qc.cz(i, i + 1)
        else:  # Default 1D
            for i in range(self.num_qubits - 1):
                qc.cz(i, i + 1)
        logger.debug(
            f"Created {self.num_qubits}-qubit cluster state, linking qubits for hypergraph correlations "
            f"(lattice: {'2d' if 'lattice' in self.custom_params and self.custom_params['lattice'].lower() == '2d' else '1d'})"
        )
        return qc


# State factory for easy instantiation
STATE_CLASSES = {
    "GHZ": GHZState,
    "W": WState,
    "CLUSTER": ClusterState,
}


def prepare_state(
    state_type: str,
    num_qubits: int,
    custom_params: Optional[Dict] = None,
) -> QuantumCircuit:
    """
    Factory function to prepare different quantum states, with configurable parameters for research.

    Supports custom parameters for experimental customization (e.g., hypergraph data, lattice type)
    and logs entanglement structure for hypergraph or fluid dynamics analysis.

    Args:
        state_type (str): Type of quantum state ("GHZ", "W", "CLUSTER").
        num_qubits (int): Number of qubits in the state (minimum 1, or 2 for W/CLUSTER).
        custom_params (dict, optional): Custom parameters for state customization
            (e.g., {"lattice": "2d", "hypergraph_data": {"entanglement": [...]}}, {"custom_gates": [...]}}).

    Returns:
        QuantumCircuit: Prepared quantum circuit.

    Raises:
        ValueError: If state_type is invalid, parameters are inconsistent, or preparation fails.
    """
    try:
        if state_type not in STATE_CLASSES:
            raise ValueError(
                f"Invalid STATE_TYPE: {state_type}. Choose from {list(STATE_CLASSES.keys())}"
            )

        # Handle custom parameters for extensibility (e.g., hypergraph data, lattice, custom gates)
        if custom_params:
            logger.debug(f"Applied custom parameters for state: {custom_params}")
            if "hypergraph_data" in custom_params:
                logger.debug(
                    f"Logging entanglement data for hypergraph: {custom_params['hypergraph_data']}"
                )
            if "custom_gates" in custom_params:
                qc = QuantumCircuit(num_qubits)
                for gate, params in custom_params["custom_gates"].items():
                    qc.append(gate, params["qargs"], params.get("cargs", []))
                return qc
        state = STATE_CLASSES[state_type](num_qubits, custom_params=custom_params)
        qc = state.create()
        return qc
    except Exception as e:
        logger.error(f"State preparation failed: {str(e)}")
        raise ValueError(f"State preparation failed: {str(e)}")


# Example config for batch runs (optional, can be loaded from JSON/YAML)
STATE_CONFIG = {
    "GHZ": {"num_qubits": 3},
    "W": {"num_qubits": 3},
    "CLUSTER": {"num_qubits": 4, "lattice": "1d"},
}
