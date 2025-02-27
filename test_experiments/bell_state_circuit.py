from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error
)
import matplotlib.pyplot as plt

# Step 1: Create the Bell State Circuit
qc = QuantumCircuit(2, 2)
qc.h(0)        # Hadamard on qubit 0
qc.cx(0, 1)    # CNOT: Entangle qubits
qc.measure([0, 1], [0, 1])  # Measure both qubits

# Step 2: Define the Noise Model
noise_model = NoiseModel()

# Add depolarizing noise to single-qubit gates
single_qubit_error = depolarizing_error(0.01, 1)  # 1% chance of random error
noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h', 'x', 'u3'])

# Add depolarizing noise to two-qubit gates (e.g., CNOT)
two_qubit_error = depolarizing_error(0.02, 2)  # 2% chance of random error
noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])

# Add amplitude damping (simulating energy loss)
single_amplitude_damping = amplitude_damping_error(0.02)  # 2% chance of energy loss
noise_model.add_all_qubit_quantum_error(single_amplitude_damping, ['h'])

# Step 3: Simulate the Circuit with Noise
backend = Aer.get_backend('qasm_simulator')
circuit_compiled = transpile(qc, backend)
job = backend.run(circuit_compiled, shots=1024, noise_model=noise_model)
result = job.result()
counts = result.get_counts()

# Step 4: Visualize the Results
print("Measurement counts with noise:", counts)
plot_histogram(counts)
plt.show()
