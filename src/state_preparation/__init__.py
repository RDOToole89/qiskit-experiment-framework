# src/state_preparation/__init__.py

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

from .ghz_state import GHZState
from .w_state import WState
from .cluster_state import ClusterState
from .base_state import BaseState
from .state_factory import prepare_state, STATE_CLASSES

__all__ = [
    "GHZState",
    "WState",
    "ClusterState",
    "BaseState",
    "prepare_state",
    "STATE_CLASSES",
]
