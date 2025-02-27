# src/state_preparation/state_constants.py

from .ghz_state import GHZState
from .w_state import WState
from .cluster_state import ClusterState

STATE_CLASSES = {
    "GHZ": GHZState,
    "W": WState,
    "CLUSTER": ClusterState,
}
