# src/config/experiment_config.py

"""
📌 Default Quantum Experiment Parameters
"""

# Default quantum experiment parameters
DEFAULT_NUM_QUBITS = 3  # Minimum: 1 (typically 2+ for entanglement)
DEFAULT_STATE_TYPE = "GHZ"  # Options: ["GHZ", "W", "CLUSTER"]
DEFAULT_NOISE_TYPE = "DEPOLARIZING"  # Options: ["DEPOLARIZING", "PHASE_FLIP", etc.]
DEFAULT_NOISE_ENABLED = True  # Enable noise in simulations?
DEFAULT_SHOTS = 1024  # Number of measurements per execution
DEFAULT_SIM_MODE = "qasm"  # ["qasm" (counts) | "density" (full quantum state)]

# Default experiment settings
DEFAULT_EXPERIMENT_NAME = "unnamed_experiment"
DEFAULT_STATEVECTOR_DIR = "results/statevector/"
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S"
