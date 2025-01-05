from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Step 1: Create a circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Step 2: Create entanglement
qc.h(0)       # Apply Hadamard to the first qubit
qc.cx(0, 1)   # Apply CNOT: entangle qubit 0 (control) with qubit 1 (target)

# Step 3: Measure both qubits
qc.measure(0, 0)  # Measure qubit 0 into classical bit 0
qc.measure(1, 1)  # Measure qubit 1 into classical bit 1

# Step 4: Simulate the circuit using Aer simulator
backend = Aer.get_backend('aer_simulator')
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

# Step 5: Visualize the results
print("Measurement counts:", counts)
plot_histogram(counts)
plt.show()
