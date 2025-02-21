# src/config/noise_config.py

"""
📌 Noise Configuration

Defines error probabilities and relaxation times for quantum simulations.
"""

DEFAULT_ERROR_RATE = 0.1  # General error probability (depolarizing, bit flip, etc.)
DEFAULT_T1 = 100e-6  # T1 relaxation time for thermal noise (seconds)
DEFAULT_T2 = 80e-6  # T2 dephasing time for thermal noise (seconds)
DEFAULT_Z_PROB = 0.5  # Probability of phase flip (Z gate) for PHASE_FLIP noise
DEFAULT_I_PROB = 0.5  # Probability of identity (no error) for PHASE_FLIP noise
