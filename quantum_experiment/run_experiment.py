# src/quantum_experiment/run_experiment.py

"""
Run quantum experiments with specified parameters.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from quantum_experiment.state_preparation import prepare_state
from quantum_experiment.noise_models import create_noise_model


def run_experiment(
    num_qubits,
    state_type="GHZ",
    noise_type="DEPOLARIZING",
    noise_enabled=True,
    shots=1024,
    sim_mode="qasm",
):
    """Runs a quantum experiment with specified parameters."""
    qc = prepare_state(state_type, num_qubits)

    # ✅ Pass `num_qubits` to `create_noise_model`
    noise_model = create_noise_model(noise_type, num_qubits) if noise_enabled else None

    backend = (
        AerSimulator(method="density_matrix")
        if sim_mode == "density"
        else Aer.get_backend("qasm_simulator")
    )

    if sim_mode == "qasm":
        # ✅ Explicitly add classical bits for measurement
        qc.add_register(QuantumCircuit(num_qubits, num_qubits).cregs[0])
        qc.measure(range(num_qubits), range(num_qubits))

    circuit_compiled = transpile(qc, backend)
    job = backend.run(circuit_compiled, shots=shots, noise_model=noise_model)
    return job.result()