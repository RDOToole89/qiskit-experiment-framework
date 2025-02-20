from qiskit import QuantumCircuit
import numpy as np


def create_ghz_state(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def create_w_state(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    w_state = np.zeros(2**num_qubits, dtype=complex)
    for i in range(num_qubits):
        w_state[1 << i] = 1 / np.sqrt(num_qubits)
    qc.initialize(w_state, range(num_qubits))
    return qc


def create_gcry_state(num_qubits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.x(0)
    for i in range(num_qubits - 1):
        qc.cry(2.0944, i, i + 1)
    return qc


def prepare_state(state_type, num_qubits):
    if state_type == "GHZ":
        return create_ghz_state(num_qubits)
    elif state_type == "W":
        return create_w_state(num_qubits)
    elif state_type == "G-CRY":
        return create_gcry_state(num_qubits)
    else:
        raise ValueError("Invalid STATE_TYPE")
