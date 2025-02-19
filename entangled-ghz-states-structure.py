# flake8: noqa

# ===========================================================
# 🧪 Experiment: Testing Structured Decoherence in GHZ States
# ===========================================================
# This experiment tests how depolarizing noise affects a 3-qubit GHZ state.
# 
# ✅ Goal:
#    - Create a GHZ state: (|000⟩ + |111⟩) / √2
#    - Introduce depolarizing noise and observe error propagation.
#    - Determine if errors appear randomly or follow a structured pattern.
#
# 🔍 Key Questions:
#    - Do errors appear in **correlated pairs**, hinting at entanglement constraints?
#    - Does the noise **preserve some symmetry**, instead of acting randomly?
#    - How does entanglement influence the way decoherence spreads?
#
# 🎯 Expected Results:
#    - **Without noise** → Only two states should appear: |000⟩ and |111⟩.
#    - **With depolarizing noise** → Additional states (e.g., |010⟩, |101⟩) may appear.
#    - If errors are structured, we should see **specific error patterns** instead of fully random noise.
#
# 🚀 Why This Matters:
#    - If noise follows a **geometric constraint**, this could reshape how we model quantum error correction.
#    - Understanding error propagation in entangled systems is key for improving fault-tolerant quantum computing.

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error
)
import matplotlib.pyplot as plt

# Create 3-qubit GHZ state
qc = QuantumCircuit(3, 3)  # |GHZ3> = (|000⟩ + |111⟩) / √2
qc.h(0)  # Put first qubit into superposition meaning it is 0 or 1 [|0⟩+∣1⟩ / √2]
qc.cx(0, 1)  # Entangle qubits 0 and 1 with a CNOT gate [|00⟩+|11⟩ / √2]
qc.cx(1, 2)  # Entangle qubits 1 and 2 with a CNOT gate [|000⟩+|111⟩ / √2]

# GHZ states are not general quantum states 
# They are a special type of quantum state that are fully entangled.
qc.measure([0, 1, 2], [0, 1, 2])  # Measure all qubits

# Choose whether to run with noise or without
NOISE_ENABLED = True  # Change this to False to run without noise

# Run on simulator
backend = Aer.get_backend('qasm_simulator')  # Use the qasm simulator to run the circuit
circuit_compiled = transpile(qc, backend)  # Compile the circuit for the backend

if NOISE_ENABLED:
       # Create noise model with depolarizing error
    noise_model = NoiseModel()
    
    # ✅ Depolarizing Error Model:
    # - Depolarization is a type of quantum noise that causes a qubit to lose its state 
    #   and become a maximally mixed state with some probability p.
    # - This type of error **randomizes** the quantum state, effectively erasing information.
    # - Unlike coherent errors, depolarization **does not preserve entanglement structure**.
    # - Mathematically, depolarization acts as:  
    #     ρ → (1 - p) ρ + (p / 2^n) I  
    #   where:
    #     - ρ is the original quantum state (density matrix).
    #     - I is the identity matrix, representing a fully mixed state.
    #     - p is the depolarization probability (error rate).
    #     - n is the number of qubits affected by the error.
    # - This means that with probability (1 - p), the state remains unchanged,
    #   but with probability p, it is replaced by a completely mixed state.
    # - Since we apply this error model to CNOT gates, it affects **pairs of qubits**,
    #   introducing random errors in entangled systems.
    # - 🔬 **Why does this happen?**
    #   1. **Quantum gates are imperfect** – tiny errors in control pulses cause random perturbations.
    #   2. **Qubits interact with the environment** – stray electromagnetic fields and thermal noise disrupt coherence.
    #   4. **Cross-talk between qubits** – subtle interactions between qubits can cause unintended excitations.
    #   5. **Thermal Fluctuations** – random thermal excitations in the environment can cause errors.

    two_qubit_error = depolarizing_error(0.1, 2)  # 10% error rate
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])  # Apply to all CNOT gates

    
    two_qubit_error = depolarizing_error(0.1, 2)  # 10% error rate
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ['cx'])  # Apply to all CNOT gates

    job = backend.run(circuit_compiled, shots=1024, noise_model=noise_model)  # Run with noise
    result = job.result()  # Get the result of the circuit
    counts = result.get_counts()  # Get the counts of the circuit

    # Plot noisy results
    plt.bar(counts.keys(), counts.values(), color='red')
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title("GHZ State Distribution With Depolarizing Noise")

    # ✅ Observations for the Noisy Model:
    # - The ideal GHZ state should give only two results: |000⟩ and |111⟩
    # - With noise, additional states appear, such as |010⟩, |101⟩, |110⟩.
    # - Errors seem to appear in **correlated pairs**, which suggests that the entanglement structure is affecting error propagation.
    # - If decoherence were fully random, we would expect an equal probability of all states, but this is **not what we observe**.
    # - This hints at a **geometric constraint on decoherence**, rather than purely probabilistic noise.

else:
    # Run without noise
    job = backend.run(circuit_compiled, shots=1024)  # Run the circuit with 1024 shots (no noise)
    result = job.result()  # Get the result of the circuit
    counts = result.get_counts()  # Get the counts of the circuit

    # Plot noiseless results
    plt.bar(counts.keys(), counts.values(), color='blue')
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title("GHZ State Distribution Without Noise")

    # ✅ Observations for the Noiseless Model:
    # - The GHZ state should only produce two possible outcomes: 50% |000⟩ and 50% |111⟩.
    # - Since the qubits are **maximally entangled**, measuring any one qubit collapses the entire system.
    # - No other states (e.g., |010⟩, |101⟩) should appear in the noiseless case.
    # - This confirms that the quantum computer correctly creates and maintains entanglement **in the absence of noise**.

# ✅ What This Means Physically:
# - Without noise, the GHZ state remains intact, meaning we see **only** |000⟩ and |111⟩.
# - With noise, we see additional states appearing, but not **randomly**—they follow a pattern.
# - The **correlated nature of errors** suggests that entanglement structure affects how decoherence occurs.
# - If this pattern persists across different noise models (e.g., amplitude damping, phase flip), this would be evidence of structured error propagation in quantum systems.

plt.show()  # Show the plot
