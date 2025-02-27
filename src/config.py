# src/quantum_experiment/config.py

"""
Configuration file for default experiment parameters.

This module centralizes all default settings for quantum experiments, ensuring:
- Consistent default values across interactive mode, CLI, and scripts.
- Easy modification for new state types, noise models, and research goals.
- Scalability for larger experiments, new backends, and hypergraph analysis.

🔹 Default Parameters:
- **Number of qubits**: Controls system size.
- **Quantum states**: Supports GHZ, W, CLUSTER (configurable lattice).
- **Noise models**: Includes DEPOLARIZING, PHASE_FLIP, THERMAL_RELAXATION, etc.
- **Simulation modes**: Choose `qasm` (counts) or `density` (full quantum state).
- **Error parameters**: Customizable noise levels for structured decoherence studies.
"""

# === ✅ Default Experiment Parameters ===
DEFAULT_NUM_QUBITS = 3  # Minimum 1, typically 2+ for entanglement states
DEFAULT_STATE_TYPE = "GHZ"  # Options: "GHZ", "W", "CLUSTER"
DEFAULT_NOISE_TYPE = (
    "DEPOLARIZING"  # Options: "DEPOLARIZING", "PHASE_FLIP", "BIT_FLIP", etc.
)
DEFAULT_NOISE_ENABLED = True  # Enable noise in simulations?
DEFAULT_SHOTS = 1024  # Number of measurements per execution
DEFAULT_SIM_MODE = "qasm"  # Options: "qasm" (counts) or "density" (full quantum state)

# === 🔧 Default Noise Parameters (Can be overridden via CLI) ===
DEFAULT_ERROR_RATE = (
    0.1  # Generic error probability (used in depolarizing, bit flip, etc.)
)
DEFAULT_T1 = 100e-6  # Relaxation time for thermal noise (seconds)
DEFAULT_T2 = 80e-6  # Dephasing time for thermal noise (seconds)
DEFAULT_Z_PROB = 0.5  # Probability of phase flip (Z gate) for PHASE_FLIP noise
DEFAULT_I_PROB = 0.5  # Probability of identity (no error) for PHASE_FLIP noise

# === 🎛 Default State Parameters ===
DEFAULT_CLUSTER_LATTICE = "2D"  # Options: "2D", "3D", "4D", etc.

# === 📂 File & Logging Configurations ===
DEFAULT_RESULTS_DIR = "results"  # Where experiment outputs are stored
DEFAULT_LOGS_DIR = "logs"  # Log directory for debugging
DEFAULT_LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
