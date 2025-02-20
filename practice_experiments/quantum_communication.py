from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

import matplotlib.pyplot as plt

# Function to create and simulate the quantum communication circuit
def quantum_communication(eavesdrop=False):
    # Step 1: Create the Bell state (entanglement) shared between Alice and Bob
    qc = QuantumCircuit(2, 2)  # 2 qubits, 2 classical bits
    qc.h(0)  # Hadamard gate on Alice's qubit
    qc.cx(0, 1)  # CNOT gate entangling Alice's and Bob's qubits

    # Optional: Simulate eavesdropping
    if eavesdrop:
        # Eve (eavesdropper) measures Alice's qubit, disturbing the entanglement
        qc.measure(0, 0)
        qc.barrier()  # Barrier for clarity

    # Step 2: Alice performs her measurement
    qc.measure(0, 0)

    # Step 3: Bob measures his qubit
    qc.measure(1, 1)

    # Step 4: Simulate the circuit
    backend = Aer.get_backend('qasm_simulator')
    job = transpile(qc, backend)
    result = backend.run(job, shots=1024).result()
    counts = result.get_counts()

    return counts, qc

# Run the simulation without eavesdropping
counts_ideal, qc_ideal = quantum_communication(eavesdrop=False)
print("Results without eavesdropping:", counts_ideal)

# Plot the results
plot_histogram(counts_ideal, title="Results Without Eavesdropping")
plt.show()

# Run the simulation with eavesdropping
counts_eavesdrop, qc_eavesdrop = quantum_communication(eavesdrop=True)
print("Results with eavesdropping:", counts_eavesdrop)

# Plot the results
plot_histogram(counts_eavesdrop, title="Results With Eavesdropping")
plt.show()
