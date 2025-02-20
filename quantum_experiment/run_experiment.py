import sys
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import Aer, AerSimulator
from quantum_experiment.state_preparation import prepare_state
from quantum_experiment.noise_models import create_noise_model


def run_experiment(num_qubits, state_type, noise_type, noise_enabled, shots, sim_mode):
    qc = prepare_state(state_type, num_qubits)

    if noise_enabled:
        noise_model = create_noise_model(noise_type)
    else:
        noise_model = None

    if sim_mode == "density":
        backend = AerSimulator(method="density_matrix")
    else:
        backend = Aer.get_backend("qasm_simulator")
        qc.measure(range(num_qubits), range(num_qubits))

    circuit_compiled = transpile(qc, backend)

    job = backend.run(circuit_compiled, shots=shots, noise_model=noise_model)
    result = job.result()

    if sim_mode == "density":
        density_matrix = result.data(0)["density_matrix"]
        return density_matrix
    else:
        counts = result.get_counts()
        return counts


if __name__ == "__main__":
    num_qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    state_type = sys.argv[2].upper() if len(sys.argv) > 2 else "GHZ"
    noise_type = sys.argv[3].upper() if len(sys.argv) > 3 else "DEPOLARIZING"
    noise_enabled = sys.argv[4].strip().lower() == "true" if len(sys.argv) > 4 else True
    shots = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    sim_mode = sys.argv[6].lower() if len(sys.argv) > 6 else "qasm"

    result = run_experiment(
        num_qubits, state_type, noise_type, noise_enabled, shots, sim_mode
    )
    print("Experiment result:", result)
