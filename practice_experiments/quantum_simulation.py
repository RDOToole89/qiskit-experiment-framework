from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Step 1: Create a circuit with 4 qubits and 4 classical bits
qc = QuantumCircuit(4, 4)

# Step 2: Apply Hadamard gate to all qubits to create superposition
for i in range(4):
    qc.h(i)

# Step 3: Measure all qubits into their corresponding classical bits
for i in range(4):
    qc.measure(i, i)

# Step 4: Simulate the circuit using Aer simulator
backend = Aer.get_backend('aer_simulator')
job = backend.run(qc, shots=1024)
result = job.result()
counts = result.get_counts()

# Step 5: Visualize the results
print("Measurement counts:", counts)
plot_histogram(counts)
plt.show()
