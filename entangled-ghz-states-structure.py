# flake8: noqa

# ===========================================================
# 🧪 Experiment: Testing Structured Decoherence in GHZ or W States
# ===========================================================
# This experiment tests how different types of quantum noise affect a 3 or more qubit GHZ or W states.
#
# ✅ Goals:
#    - Create a GHZ state: (|000⟩ + |111⟩) / √2 (or other states like W or G-CRY).
#    - Introduce various types of quantum noise: depolarizing, phase flip, amplitude damping, or phase damping.
#    - Observe error propagation and determine if errors follow structured patterns instead of being purely random.
#
# 🔍 Key Questions:
#    - Do errors appear in correlated pairs, hinting at entanglement constraints?
#    - Does the noise preserve some symmetry rather than acting randomly?
#    - How does entanglement influence the way decoherence spreads?
#
# 🎯 Expected Results:
#    - Without noise: The ideal state (e.g., GHZ) should produce only two outcomes: |000⟩ and |111⟩.
#    - With noise: Additional states or altered coherences appear, with differences based on the noise type.
#
# 🚀 Why This Matters:
#    - If noise follows a geometric constraint, it could reshape models for quantum error correction.
#    - Understanding how entangled states decay under various noise channels is critical for developing
#      fault-tolerant quantum computing techniques.
#
# 🛠 Additional Functionality:
#    - The experiment now accepts a SIM_MODE parameter:
#         * "qasm": Uses the qasm_simulator, including measurement operations to yield a classical
#           distribution (histogram) of outcomes.
#         * "density": Uses the density matrix simulation method (via AerSimulator) to output the full density matrix of the state.
#    - An extra noise type, PHASE_DAMPING, has been added to model dephasing effects that reduce
#      the off-diagonal elements of the state without altering populations.
#
# 📖 Usage:
#    python <script> NUM_QUBITS STATE_TYPE NOISE_TYPE NOISE_ENABLED SHOTS SIM_MODE
#
#    - NUM_QUBITS: Number of qubits (default 3).
#    - STATE_TYPE: "GHZ", "W", or "G-CRY" (default GHZ).
#    - NOISE_TYPE: "DEPOLARIZING", "PHASE_FLIP", "AMPLITUDE_DAMPING", or "PHASE_DAMPING" (default DEPOLARIZING).
#    - NOISE_ENABLED: "true" or "false" (default true).
#    - SHOTS: Number of shots (default 1024).
#    - SIM_MODE: "qasm" for measurement-based simulation or "density" for full density matrix output (default "qasm").
#
#    Example:
#       python script.py 3 GHZ DEPOLARIZING false 1024 density
#
# ✅ Notes:
#    - When using SIM_MODE = "density", measurement operations are removed and a save instruction is added so that the full state (density matrix)
#      can be observed. This is useful for analyzing coherence and mixedness directly.
#    - For measurement-based simulations (SIM_MODE = "qasm"), the circuit includes measurement gates, and you will
#      see a histogram of outcomes.
#
# ===========================================================

import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
    phase_damping_error,
)
import matplotlib.pyplot as plt

# ✅ Read command-line arguments
# Usage: python <script> NUM_QUBITS STATE_TYPE NOISE_TYPE NOISE_ENABLED SHOTS SIM_MODE
NUM_QUBITS = int(sys.argv[1]) if len(sys.argv) > 1 else 3  # Default to 3 qubits
STATE_TYPE = sys.argv[2].upper() if len(sys.argv) > 2 else "GHZ"  # GHZ, W, G-CRY
NOISE_TYPE = sys.argv[3].upper() if len(sys.argv) > 3 else "DEPOLARIZING"
NOISE_ENABLED = sys.argv[4].strip().lower() == "true" if len(sys.argv) > 4 else True
SHOTS = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
SIM_MODE = sys.argv[6].lower() if len(sys.argv) > 6 else "qasm"  # "qasm" or "density"

print(
    f"Running with state type: {STATE_TYPE}, noise type: {NOISE_TYPE}, Noise Enabled: {NOISE_ENABLED}, Simulation Mode: {SIM_MODE}"
)

# ✅ Create N-qubit circuit
qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)

if STATE_TYPE == "GHZ":
    qc.h(0)
    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)
elif STATE_TYPE == "W":
    # Initialize an array for the correct W-state
    w_state = np.zeros(2**NUM_QUBITS, dtype=complex)
    # Set amplitudes: each basis state with a single '1' gets equal amplitude.
    for i in range(NUM_QUBITS):
        w_state[1 << i] = 1 / np.sqrt(NUM_QUBITS)
    qc.initialize(w_state, range(NUM_QUBITS))
elif STATE_TYPE == "G-CRY":
    qc.x(0)
    for i in range(NUM_QUBITS - 1):
        qc.cry(2.0944, i, i + 1)  # CRY(pi/3) to create superposition
else:
    raise ValueError(f"Invalid STATE_TYPE: {STATE_TYPE}. Choose 'GHZ', 'W', or 'G-CRY'.")

# For noise types that require a basis change, apply Hadamard gates on all qubits.
if NOISE_ENABLED and NOISE_TYPE in ["PHASE_FLIP"]:
    qc.h(range(NUM_QUBITS))
if NOISE_ENABLED and NOISE_TYPE in ["AMPLITUDE_DAMPING"]:
    qc.h(range(NUM_QUBITS))

# Depending on simulation mode, add measurement operations or remove them.
if SIM_MODE == "qasm":
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
else:
    # For density matrix simulation, remove measurements
    qc = qc.remove_final_measurements(inplace=False)
    # And add a save instruction to save the density matrix.
    qc.save_density_matrix()

# --------------------------------------
# Debugging: Get the ideal statevector (without noise or measurement)
if SIM_MODE == "qasm":
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    ideal_state = Statevector.from_instruction(qc_no_meas)
    print("Ideal statevector (before noise & measurement):")
    print(ideal_state)
else:
    print("Skipping statevector debugging in density mode.")

# Choose backend based on SIM_MODE
if SIM_MODE == "density":
    backend = AerSimulator(method="density_matrix")
else:
    backend = Aer.get_backend("qasm_simulator")

circuit_compiled = transpile(qc, backend)

# ✅ Build noise model if noise is enabled
noise_model = None
if NOISE_ENABLED:
    print(f"Applying {NOISE_TYPE} noise to the system...")
    noise_model = NoiseModel()

    if NOISE_TYPE == "DEPOLARIZING":
        noise = depolarizing_error(0.2, 2)
        noise_model.add_all_qubit_quantum_error(noise, ["cx"])
    elif NOISE_TYPE == "PHASE_FLIP":
        noise = pauli_error([("Z", 0.1), ("I", 0.9)])
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])
    elif NOISE_TYPE == "AMPLITUDE_DAMPING":
        noise = amplitude_damping_error(0.1)
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])
    elif NOISE_TYPE == "PHASE_DAMPING":
        noise = phase_damping_error(0.1)
        noise_model.add_all_qubit_quantum_error(noise, ["id", "u1", "u2", "u3"])
    else:
        raise ValueError(
            f"Invalid NOISE_TYPE: {NOISE_TYPE}. Choose 'DEPOLARIZING', 'PHASE_FLIP', 'AMPLITUDE_DAMPING', or 'PHASE_DAMPING'."
        )

print(f"Running with NOISE_ENABLED={NOISE_ENABLED}, Noise Model={noise_model}")

# ✅ Run the simulation based on SIM_MODE
if SIM_MODE == "density":
    job = backend.run(circuit_compiled, noise_model=noise_model)
    result = job.result()
    # Extract the density matrix from the result data
    density_matrix = result.data(0)["density_matrix"]
    print("Final Density Matrix:")
    print(density_matrix)
    # Draw a text-based representation of the density matrix
    # print(density_matrix.draw("text"))
    # Create a heatmap visualization for a more intuitive look
    dm_array = np.abs(density_matrix.data)
    plt.figure(figsize=(8, 6))
    plt.imshow(dm_array, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Magnitude")
    plt.title("Density Matrix Heatmap (Absolute Values)")
    plt.xlabel("Basis State Index")
    plt.ylabel("Basis State Index")
    plt.show()
    # Print LaTeX representation of the density matrix
    # print("\nLaTeX representation:\n")
    # print(density_matrix.draw("latex"))
else:
    job = backend.run(circuit_compiled, shots=SHOTS, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts()
    # ✅ Plot results
    color = "red" if NOISE_ENABLED else "blue"
    title = f"{STATE_TYPE} State Distribution {'With ' + NOISE_TYPE + ' Noise' if NOISE_ENABLED else 'Without Noise'}"
    plt.bar(counts.keys(), counts.values(), color=color)
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.show()


# ✅ Observations:
# - Without noise: The GHZ state should only produce |000⟩ and |111⟩ in equal probability.
# - With depolarizing noise: Additional states appear due to random bit flips.
# - With phase flip noise: The phase of the quantum state is affected, influencing interference patterns.
# - If noise is structured rather than random, this suggests an underlying constraint in decoherence.

# ✅ Why This Matters:
# - If decoherence patterns are **not completely random**, it could mean entanglement imposes geometric constraints.
# - Investigating these structured patterns could improve **quantum error correction** methods.
# - Further experiments could explore **other noise types** (amplitude damping, thermal noise) to see if they follow similar patterns.

# ✅ Observations of Depolarizing Noise:
# - The expected GHZ distribution (50% |000⟩, 50% |111⟩) **breaks down**, and we see other states appearing.
# - The **errors seem to occur in correlated pairs** (e.g., |010⟩, |101⟩, |110⟩), rather than a fully random spread.
# - This suggests that even though depolarizing noise is **random at the gate level**, the entanglement structure still influences how errors propagate.
# - If the noise were truly random, we would expect a fully uniform distribution of all possible states.
# - Instead, the structure of entanglement appears to **bias the probability of errors**, meaning **decoherence might follow deeper constraints**.

# ✅ Observations of Phase Flip Noise:
# - Before applying Hadamard gates, the state remains **stuck** in |000⟩ and |111⟩, since phase errors do not affect measurement in the computational basis.
# - After applying Hadamard gates, **the expected GHZ pattern disappears** and is replaced by an equal mix of |000⟩, |011⟩, |101⟩, and |110⟩.
# - This confirms that **phase flip errors transform into bit flips** when measured in the Hadamard basis.
# - **No single state dominates**, suggesting that phase flips affect all components of the superposition equally.
# - If phase flip errors were purely random, we might expect a fully uniform distribution across all 8 possible states, but **only 4 dominate**, hinting at a structured transformation of errors.
# - The symmetry of error distribution indicates that **entanglement plays a role in error propagation**, even when dealing with phase errors.

# ✅ Observations of Amplitude Damping Noise:
# - **Without Hadamard transformation:**
#   - The GHZ state was **mostly preserved** because **amplitude damping primarily affects qubits in |1⟩**.
#   - There was a **slight bias toward |0⟩**, meaning the system **prefers the ground state**.
# - **After applying Hadamards:**
#   - The GHZ pattern **decomposed** into **four states**:
#     **|000⟩, |011⟩, |101⟩, |110⟩**.
#   - **Same even-parity structure as phase flip noise**, but with a slight **bias toward |0⟩**.
# - **Key Feature:** Unlike depolarizing or phase flip noise, **amplitude damping represents irreversible energy loss**, which leads to a **slow drift toward the ground state**.
# - **The same parity constraints still hold**, meaning **error propagation is structured, not purely random**.


# ✅ Observations of Amplitude Damping Noise vs Phase Flip Noise:
# - Both noise types **preserve even parity**, meaning errors **do not spread randomly across all 8 states**.
# - **Phase flip noise:**
#   - Only **affects phase relationships** but does **not cause energy loss**.
#   - Hadamard transformation converts these **hidden errors into measurable bit flips**.
# - **Amplitude damping noise:**
#   - Represents **energy decay** rather than just phase shifts.
#   - Causes a **slight bias toward |0⟩**, meaning **states closer to the ground state become more probable**.
# - **Both noise models result in a nearly even spread** among **4 states (|000⟩, |011⟩, |101⟩, |110⟩)**, showing that **entanglement constrains how errors propagate**.

# ✅ Latest Findings: W-State Error Propagation
# - We tested the effect of **phase flip noise** and **depolarizing noise** on the W-state.
# - The W-state showed structured error propagation, similar to GHZ states, but with key differences.

# ✅ Observations of Depolarizing Noise on the W-State:
# - Unlike GHZ states, the W-state maintains **a uniform distribution** among |100⟩, |010⟩, and |001⟩.
# - This suggests that **depolarization does not fully disrupt the W-state’s structure**, but spreads the noise evenly.
# - **Key takeaway**: The W-state appears more **robust** to depolarization than GHZ states, as entanglement is distributed across multiple qubits.

# ✅ Observations of Phase Flip Noise on the W-State:
# - The W-state exhibits a strong bias toward **|000⟩ and |111⟩**, with error pairs appearing in specific patterns.
# - This suggests that **phase errors introduce decoherence in a structured way**, similar to GHZ states.
# - The pattern indicates that **entanglement constraints still govern the way phase errors propagate**.
# - **Key takeaway**: The W-state does not resist phase flip errors as strongly as depolarizing noise but still follows structured decoherence.

# ✅ Key Insights from the Experiments:
# - **Entanglement constrains noise propagation** in both GHZ and W-states, leading to structured error patterns.
# - The W-state is **more resilient to depolarization** but **still sensitive to phase flips**.
# - **Further research:** Investigate how **amplitude damping noise** affects W-states to determine energy loss behavior.
# - These results could help **optimize quantum error correction methods** that exploit entanglement structure.


# ✅ What This Could Mean:
# - **Entanglement does not simply "store" quantum information—it also dictates how errors spread through the system**.
# - **Quantum noise might not be purely random**, but instead follow **hidden constraints** dictated by the underlying quantum correlations.
# - These results suggest that we could explore **error correction techniques that exploit entanglement structure**, rather than treating errors as purely probabilistic.
# - Testing this across **different types of noise** (e.g., amplitude damping, thermal noise) could confirm whether decoherence is **geometrically constrained**.
