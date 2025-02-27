# src/quantum_experiment/noise_models/__init__.py
"""
Noise models package for quantum experiments.

This package includes:
- BaseNoise class for all noise models.
- Specific noise classes (Depolarizing, Phase Flip, etc.).
- A factory function (create_noise_model) to instantiate noise models.
"""

from .base_noise import BaseNoise
from .depolarizing import DepolarizingNoise
from .phase_flip import PhaseFlipNoise
from .amplitude_damping import AmplitudeDampingNoise
from .phase_damping import PhaseDampingNoise
from .thermal_relaxation import ThermalRelaxationNoise
from .bit_flip import BitFlipNoise
from .noise_factory import create_noise_model, NOISE_CLASSES, NOISE_CONFIG

__all__ = [
    "BaseNoise",
    "DepolarizingNoise",
    "PhaseFlipNoise",
    "AmplitudeDampingNoise",
    "PhaseDampingNoise",
    "ThermalRelaxationNoise",
    "BitFlipNoise",
    "create_noise_model",
    "NOISE_CLASSES",
    "NOISE_CONFIG",
]
