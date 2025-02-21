# src/state_preparation/state_factory.py

from qiskit import QuantumCircuit
from typing import Optional, Dict
from src.config.logging_config import logger
from src.config.state_config import STATE_CLASSES, VALID_STATE_TYPES

"""
📌 State Factory Module

Provides a factory function to prepare different quantum states with configurable parameters.
"""


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
        custom_params (dict, optional): Custom parameters for state customization.

    Returns:
        QuantumCircuit: Prepared quantum circuit.

    Raises:
        ValueError: If state_type is invalid, parameters are inconsistent, or preparation fails.
    """
    try:
        if state_type not in VALID_STATE_TYPES:
            raise ValueError(
                f"Invalid STATE_TYPE: {state_type}. Choose from {VALID_STATE_TYPES}"
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
