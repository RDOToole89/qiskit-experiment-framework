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
    def plot_histogram(*args, **kwargs):
        return plot_histogram(*args, **kwargs)

    @staticmethod
    def plot_density_matrix(*args, **kwargs):
        return plot_density_matrix(*args, **kwargs)

    @staticmethod
    def plot_hypergraph(*args, **kwargs):
        return plot_hypergraph(*args, **kwargs)
