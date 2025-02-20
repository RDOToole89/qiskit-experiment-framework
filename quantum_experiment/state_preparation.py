# src/quantum_experiment/state_preparation.py

"""
State preparation functions for quantum experiments.
"""

from qiskit import QuantumCircuit
import numpy as np


def create_ghz_state(num_qubits):
    """Creates an N-qubit GHZ state."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def create_w_state(num_qubits):
    """Creates an N-qubit W state."""
    qc = QuantumCircuit(num_qubits)
    w_state = np.zeros(2**num_qubits, dtype=complex)
    for i in range(num_qubits):
        w_state[1 << i] = 1 / np.sqrt(num_qubits)
    qc.initialize(w_state, range(num_qubits))
    return qc


def create_gcry_state(num_qubits):
    """Creates an N-qubit G-CRY state."""
    qc = QuantumCircuit(num_qubits)
    qc.x(0)
    for i in range(num_qubits - 1):
        qc.cry(2.0944, i, i + 1)
    return qc


STATE_CREATORS = {
    "GHZ": create_ghz_state,
    "W": create_w_state,
    "G-CRY": create_gcry_state,
}


def prepare_state(state_type, num_qubits):
    """Factory function to prepare different quantum states."""
    if state_type not in STATE_CREATORS:
        raise ValueError(
            f"Invalid STATE_TYPE: {state_type}. Choose from {list(STATE_CREATORS.keys())}"
        )

    return STATE_CREATORS[state_type](num_qubits)
