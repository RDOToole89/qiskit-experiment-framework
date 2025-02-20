# src/quantum_experiment/noise_models.py

"""
Noise models for quantum experiments.
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
)

# Noise functions dictionary
NOISE_FUNCTIONS = {
    "DEPOLARIZING": lambda qubits: depolarizing_error(
        0.05 * min(qubits, 2), min(qubits, 2)
    ),  # Support up to 2-qubits
    "PHASE_FLIP": lambda _: pauli_error([("Z", 0.1), ("I", 0.9)]),  # Always 1-qubit
    "AMPLITUDE_DAMPING": lambda _: amplitude_damping_error(0.1),  # Always 1-qubit
    "PHASE_DAMPING": lambda _: phase_damping_error(0.1),  # Always 1-qubit
}


def create_noise_model(noise_type, num_qubits):
    """Creates a scalable noise model that applies the correct noise based on qubit count."""
    if noise_type not in NOISE_FUNCTIONS:
        raise ValueError(
            f"Invalid NOISE_TYPE: {noise_type}. Choose from {list(NOISE_FUNCTIONS.keys())}"
        )

    noise_model = NoiseModel()

    # Apply correct noise for different gate sizes
    for qubits in range(1, min(num_qubits, 2) + 1):  # Limit to 1 and 2 qubits
        noise = NOISE_FUNCTIONS[noise_type](qubits)

        if qubits == 1:
            noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])
        elif qubits == 2:
            noise_model.add_all_qubit_quantum_error(noise, ["cx"])

    return noise_model
