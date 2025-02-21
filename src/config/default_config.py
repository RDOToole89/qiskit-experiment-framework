# src/config/default_config.py

"""
📌 Default Parameters for Quantum Experiments

Ensures:
- Consistency across interactive mode, CLI, and batch runs.
- Easy modifications for new quantum states, noise models, and backends.
- Scalability for structured decoherence studies and hypergraph analysis.
"""

DEFAULT_NUM_QUBITS = 3  # Minimum: 1 (typically 2+ for entanglement)
DEFAULT_STATE_TYPE = "GHZ"  # Options: ["GHZ", "W", "CLUSTER"]
DEFAULT_NOISE_TYPE = "DEPOLARIZING"  # Options: ["DEPOLARIZING", "PHASE_FLIP", etc.]
DEFAULT_NOISE_ENABLED = True  # Enable noise in simulations?
DEFAULT_SHOTS = 1024  # Number of measurements per execution
DEFAULT_SIM_MODE = "qasm"  # ["qasm" (counts) | "density" (full quantum state)]
