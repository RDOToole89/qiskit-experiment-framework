# flake8: noqa

# ===========================================================
# 🧪 Experiment: Testing Structured Decoherence in GHZ States
# ===========================================================
# This experiment tests how different types of quantum noise affect a 3-qubit GHZ state.
#
# ✅ Goal:
#    - Create a GHZ state: (|000⟩ + |111⟩) / √2
#    - Introduce depolarizing or phase flip noise and observe error propagation.
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

import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    amplitude_damping_error,
)
import matplotlib.pyplot as plt

# ✅ Read command-line arguments
STATE_TYPE = sys.argv[1].upper() if len(sys.argv) > 1 else "GHZ" # GHZ, W, G-CRY
NOISE_TYPE = sys.argv[2].upper() if len(sys.argv) > 2 else "DEPOLARIZING"

NOISE_ENABLED = sys.argv[3].strip().lower() == "true" if len(sys.argv) > 3 else True

print(
    f"Running with state type: {STATE_TYPE}, noise type: {NOISE_TYPE}, Noise Enabled: {NOISE_ENABLED}"
)


# ✅ Create 3-qubit state
qc = QuantumCircuit(3, 3)

if STATE_TYPE == "GHZ":
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
elif STATE_TYPE == "W":
    # ----- OPTION 1: Build the W state via gates (this produces a 4-term state)
    # qc.ry(2 * np.arccos(1 / np.sqrt(3)), 0)
    # qc.cx(0, 1)
    # qc.ry(2 * np.arccos(1 / np.sqrt(2)), 1)
    # qc.cx(1, 2)

    # ----- OPTION 2: Use initialize to guarantee the canonical single-excitation W state.
    # The canonical 3-qubit W state (in Qiskit's ordering: |q2 q1 q0>) has nonzero amplitudes in:
    #   |001> (qubit0=1), |010> (qubit1=1), and |100> (qubit2=1)
    # For the basis ordering: |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>
    w_state = [
        0,               # |000>
        1/np.sqrt(3),    # |001>
        1/np.sqrt(3),    # |010>
        0,               # |011>
        1/np.sqrt(3),    # |100>
        0,               # |101>
        0,               # |110>
        0                # |111>
    ]
    qc.initialize(w_state, [0, 1, 2])
elif STATE_TYPE == "G-CRY":
    qc.x(0)
    qc.cry(2.0944, 0, 1)  # CRY(pi/3) to create equal superposition of |100> and |010>
    qc.cry(2.0944, 1, 2)  # CRY(pi/3) to create |100> + |010> + |001>
else:
    raise ValueError(f"Invalid STATE_TYPE: {STATE_TYPE}. Choose 'GHZ' or 'W'.")

if NOISE_ENABLED and NOISE_TYPE == "PHASE_FLIP":
    # ✅ Apply Hadamard gates before measurement to reveal phase flip errors.
    # - Phase flip errors affect the **relative phase** between |000⟩ and |111⟩,
    #   but these errors are **not visible** in the computational (Z) basis.
    # - Applying Hadamard gates **transforms phase errors into bit-flip errors**,
    #   making them measurable in the standard basis.
    # - This works because the Hadamard gate swaps:
    #     |0⟩ → (|0⟩ + |1⟩) / √2
    #     |1⟩ → (|0⟩ - |1⟩) / √2
    #   meaning a phase flip (Z error) becomes a bit flip in the new basis.
    # - Without this step, phase flip errors remain hidden in measurement results.
    qc.h([0, 1, 2])  # Apply Hadamards to expose phase flip errors

if NOISE_ENABLED and NOISE_TYPE == "AMPLITUDE_DAMPING":
    # ✅ Apply Hadamard gates before measurement to expose decoherence effects.
    # - Amplitude damping noise causes energy loss, which gradually collapses the state to |000⟩.
    # - However, if we measure in the standard basis, we mostly just see |000⟩.
    # - Applying Hadamard gates **spreads the lost coherence across all qubits**,
    #   making decoherence effects **visible as bit-flip errors**.
    # - This helps us detect **whether coherence is lost gradually or instantaneously**.
    qc.h([0, 1, 2])  # Apply Hadamards to expose amplitude damping effects

qc.measure([0, 1, 2], [0, 1, 2])  # Measure the qubits in the computational basis

# --------------------------------------
# Debugging: Get the ideal statevector (without measurement)
# Remove measurement operations to see the pure state.
qc_no_meas = qc.remove_final_measurements(inplace=False)
ideal_state = Statevector.from_instruction(qc_no_meas)
print("Ideal statevector (before noise & measurement):")
print(ideal_state)
# --------------------------------------

# Run on simulator
backend = Aer.get_backend("qasm_simulator")
circuit_compiled = transpile(qc, backend)

# ✅ If noise is enabled, create the appropriate noise model
noise_model = None
if NOISE_ENABLED:
    print(f"Applying {NOISE_TYPE} noise to the system...")
    noise_model = NoiseModel()

    if NOISE_TYPE == "DEPOLARIZING":
        # ✅ Depolarizing Noise Model:
        # - Randomizes the quantum state, effectively erasing information.
        # - Unlike coherent errors, depolarization **does not preserve entanglement structure**.
        # - Mathematically: ρ → (1 - p) ρ + (p / 2^n) I
        noise = depolarizing_error(0.1, 2)
        noise_model.add_all_qubit_quantum_error(noise, ["cx"])  # Apply to CNOT gates

    elif NOISE_TYPE == "PHASE_FLIP":
        # ✅ Phase Flip Noise Model:
        # - Flips the quantum phase but does not erase information.
        # - Introduces errors via Pauli-Z flips.
        # - Mathematically: ρ → (1 - p) ρ + p ZρZ
        noise = pauli_error([("Z", 0.1), ("I", 0.9)])  # 10% phase flip error
        noise_model.add_all_qubit_quantum_error(
            noise, ["id", "u1", "u2", "u3"]
        )  # Apply to single-qubit gates

    elif NOISE_TYPE == "AMPLITUDE_DAMPING":
        # ✅ Amplitude Damping Noise Model:
        # - Represents energy loss (relaxation of |1⟩ → |0⟩), common in real qubits.
        # - Unlike depolarization, it introduces **asymmetry** by favoring |0⟩ states.
        noise = amplitude_damping_error(0.1)
        noise_model.add_all_qubit_quantum_error(
            noise, ["id", "u1", "u2", "u3"]
        )  # Apply to single-qubit gates

    else:
        raise ValueError(
            f"Invalid NOISE_TYPE: {NOISE_TYPE}. Choose 'DEPOLARIZING', 'PHASE_FLIP', or 'AMPLITUDE_DAMPING'."
        )

print(f"Running with NOISE_ENABLED={NOISE_ENABLED}, Noise Model={noise_model}")
# ✅ Run the circuit with or without noise
job = (
    backend.run(circuit_compiled, shots=1024, noise_model=noise_model)
    if NOISE_ENABLED
    else backend.run(circuit_compiled, shots=1024, noise_model=None)
)

# ✅ Get results
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

# ✅ What This Could Mean:
# - **Entanglement does not simply "store" quantum information—it also dictates how errors spread through the system**.
# - **Quantum noise might not be purely random**, but instead follow **hidden constraints** dictated by the underlying quantum correlations.
# - These results suggest that we could explore **error correction techniques that exploit entanglement structure**, rather than treating errors as purely probabilistic.
# - Testing this across **different types of noise** (e.g., amplitude damping, thermal noise) could confirm whether decoherence is **geometrically constrained**.
