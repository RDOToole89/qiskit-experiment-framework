# src/visualization/visualizer.py

from .histogram import plot_histogram
from .density_matrix import plot_density_matrix
from .hypergraph import plot_hypergraph
from .statevector_visualizer import plot_statevector

"""
📌 Quantum Visualization Module

Provides tools for plotting quantum experiment results:
- Histograms (counts)
- Density matrices
- Hypergraphs (state correlations)
- Statevector probabilities (with optional saving)
"""


class Visualizer:
    """Class for handling quantum experiment visualizations."""

    @staticmethod
    def plot_histogram(*args, **kwargs):
        return plot_histogram(*args, **kwargs)

    @staticmethod
    def plot_density_matrix(*args, **kwargs):
        return plot_density_matrix(*args, **kwargs)

    @staticmethod
    def plot_hypergraph(*args, **kwargs):
        return plot_hypergraph(*args, **kwargs)

    @staticmethod
    def plot_statevector(statevector_file: str, save_path: str = None):
        """
        Plots the statevector from a saved file and optionally saves it.

        Args:
            statevector_file (str): Path to the statevector file.
            save_path (str, optional): Path to save the plot.
        """
        return plot_statevector(statevector_file, save_path)
