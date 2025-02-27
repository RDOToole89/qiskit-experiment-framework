# src/state_preparation/state_factory.py

from qiskit import QuantumCircuit
from typing import Optional, Dict
import logging

from .ghz_state import GHZState
from .w_state import WState
from .cluster_state import ClusterState

logger = logging.getLogger("QuantumExperiment.StatePreparation")

STATE_CLASSES = {
    "GHZ": GHZState,
    "W": WState,
    "CLUSTER": ClusterState,
}

def prepare_state(
    state_type: str,
    num_qubits: int,
    custom_params: Optional[Dict] = None,
    add_barrier: bool = False,  # New parameter
) -> QuantumCircuit:
    """
    Factory function to prepare different quantum states.

    Args:
        state_type (str): "GHZ", "W", or "CLUSTER".
        num_qubits (int): Number of qubits.
        custom_params (dict, optional): Custom parameters for state customization.
        add_barrier (bool, optional): Whether to add a barrier after state preparation.

    Returns:
        QuantumCircuit: Prepared quantum circuit.
    """
    try:
        if state_type not in STATE_CLASSES:
            raise ValueError(f"Invalid state type: {state_type}. Choose from {list(STATE_CLASSES.keys())}")

        # Check for custom gates handling (if any)
        if custom_params and "custom_gates" in custom_params:
            qc = QuantumCircuit(num_qubits)
            for gate, params in custom_params["custom_gates"].items():
                qc.append(gate, params["qargs"], params.get("cargs", []))
            return qc

        state = STATE_CLASSES[state_type](num_qubits, custom_params=custom_params)
        qc = state.create(add_barrier=add_barrier)  # Pass add_barrier to create
        logger.debug(f"Successfully created {state_type} state.")
        return qc
    except Exception as e:
        logger.error(f"State preparation failed: {str(e)}")
        raise ValueError(f"State preparation failed: {str(e)}")