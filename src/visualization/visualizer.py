# src/visualization/visualizer.py

"""
Visualizer class for quantum experiments.

Provides static methods to call the individual visualization functions.
"""

from .histogram import plot_histogram
from .density_matrix import plot_density_matrix
from .hypergraph import plot_hypergraph

class Visualizer:
    @staticmethod
    def plot_histogram(*args, num_qubits=None, **kwargs):
        return plot_histogram(*args, num_qubits=num_qubits, **kwargs)

    @staticmethod
    def plot_density_matrix(*args, state_type=None, noise_type=None, **kwargs):
        return plot_density_matrix(*args, state_type=state_type, noise_type=noise_type, **kwargs)

    @staticmethod
    def plot_hypergraph(*args, state_type=None, noise_type=None, **kwargs):
        return plot_hypergraph(*args, state_type=state_type, noise_type=noise_type, **kwargs)
