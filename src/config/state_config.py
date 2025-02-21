# src/config/state_config.py

from src.state_preparation.ghz_state import GHZState
from src.state_preparation.w_state import WState
from src.state_preparation.cluster_state import ClusterState

"""
📌 State Configuration

Defines mappings for quantum states and default state settings.
"""

STATE_CLASSES = {
    "GHZ": GHZState,
    "W": WState,
    "CLUSTER": ClusterState,
}

VALID_STATE_TYPES = list(STATE_CLASSES.keys())  # List of valid state names
DEFAULT_CLUSTER_LATTICE = "2D"  # Cluster state lattice structure: ["2D", "3D", "4D"]
