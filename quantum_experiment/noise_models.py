from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
)


def create_noise_model(noise_type):
    noise_model = NoiseModel()

    if noise_type == "DEPOLARIZING":
        noise = depolarizing_error(0.2, 2)
        noise_model.add_all_qubit_quantum_error(noise, ["cx"])

    elif noise_type == "PHASE_FLIP":
        noise = pauli_error([("Z", 0.1), ("I", 0.9)])
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])

    elif noise_type == "AMPLITUDE_DAMPING":
        noise = amplitude_damping_error(0.1)
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])

    elif noise_type == "PHASE_DAMPING":
        noise = phase_damping_error(0.1)
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])

    else:
        raise ValueError("Invalid NOISE_TYPE")

    return noise_model
