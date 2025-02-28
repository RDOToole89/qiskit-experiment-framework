# src/config/constants.py

"""
Constants for the Quantum Experiment Interactive Runner.
"""

# Valid noise types and state types
VALID_NOISE_TYPES = [
    "DEPOLARIZING",
    "PHASE_FLIP",
    "AMPLITUDE_DAMPING",
    "PHASE_DAMPING",
    "THERMAL_RELAXATION",
    "BIT_FLIP",
]
VALID_STATE_TYPES = ["GHZ", "W", "CLUSTER"]

# One-letter shortcuts for noise types (case-insensitive)
NOISE_SHORTCUTS = {
    "d": "DEPOLARIZING",
    "p": "PHASE_FLIP",
    "a": "AMPLITUDE_DAMPING",
    "z": "PHASE_DAMPING",
    "t": "THERMAL_RELAXATION",
    "b": "BIT_FLIP",
}

# Single-qubit noise types
SINGLE_QUBIT_NOISE_TYPES = ["AMPLITUDE_DAMPING", "PHASE_DAMPING", "BIT_FLIP"]
