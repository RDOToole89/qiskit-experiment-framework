# src/quantum_experiment/__init__.py

"""
Quantum Experiment Framework

This package provides:
- **State preparation**: GHZ, W, CLUSTER states with entanglement control.
- **Noise modeling**: Depolarizing, phase flip, thermal relaxation, bit flip, etc.
- **Quantum execution**: Configurable simulation with Qiskit Aer.
- **Visualization tools**: Histograms, density matrices, hypergraph mapping.
- **Utilities**: Logging, input validation, experiment configuration.

Designed for modular quantum experiments, extensibility, and research integration.

🔹 Core Features:
- Supports hypergraph correlation analysis and structured decoherence studies.
- Provides CLI and interactive execution modes.
- Modular architecture for adding new noise models, states, and research tools.
"""

from .state_preparation import prepare_state
from .noise_models import create_noise_model
from .run_experiment import run_experiment
from .visualization import Visualizer  # Uses updated class-based visualization
from .config.config import (  # Centralized experiment configurations
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
    DEFAULT_ERROR_RATE,
    DEFAULT_T1,
    DEFAULT_T2,
    DEFAULT_Z_PROB,
    DEFAULT_I_PROB,
    DEFAULT_CLUSTER_LATTICE,
)

# Expose key functions and classes for easier package imports
__all__ = [
    "prepare_state",
    "create_noise_model",
    "run_experiment",
    "Visualizer",
    "ExperimentUtils",
    "DEFAULT_NUM_QUBITS",
    "DEFAULT_STATE_TYPE",
    "DEFAULT_NOISE_TYPE",
    "DEFAULT_NOISE_ENABLED",
    "DEFAULT_SHOTS",
    "DEFAULT_SIM_MODE",
    "DEFAULT_ERROR_RATE",
    "DEFAULT_T1",
    "DEFAULT_T2",
    "DEFAULT_Z_PROB",
    "DEFAULT_I_PROB",
    "DEFAULT_CLUSTER_LATTICE",
]
