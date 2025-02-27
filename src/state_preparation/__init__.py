# src/state_preparation/__init__.py

"""
State preparation package for quantum experiments.

This package provides:
- BaseState: A template for state creation.
- GHZState, WState, ClusterState: Implementations of different quantum states.
- prepare_state: A factory function for creating states.
- STATE_CLASSES: Dictionary mapping state types to their classes.
"""

from .base_state import BaseState
from .ghz_state import GHZState
from .w_state import WState
from .cluster_state import ClusterState
from .state_factory import prepare_state
from .state_constants import STATE_CLASSES

__all__ = [
    "BaseState",
    "GHZState",
    "WState",
    "ClusterState",
    "prepare_state",
    "STATE_CLASSES",
]
