# src/quantum_experiment/visualization.py

"""
Visualization functions for quantum experiments, designed for extensibility and research integration.

This module provides tools for plotting experiment results (histograms, density matrices, hypergraphs),
supporting a scalable "quantum experimental lab." Integrates with new noise models, states,
and research features (e.g., hypergraph correlations, fluid dynamics in Hilbert space).

Key features:
- Enhanced plotting for histograms (counts) and density matrices, with save options and customization.
- Hypergraph visualization for correlation analysis, tying to noise and state effects.
- Logging of visualization metadata for research (e.g., hypergraphs, fluid flow).
- Scalable for large qubit counts, with filtering or aggregation options.
- Extensible for custom visualizations, params, or backends (e.g., IBM data).

Example usage:
    Visualizer.plot_histogram(counts, state_type="GHZ", noise_type="DEPOLARIZING", save_path="histogram.png")
    Visualizer.plot_density_matrix(density_matrix, cmap="viridis", show_real=True)
    Visualizer.plot_hypergraph(correlation_data, state_type="GHZ")
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import DensityMatrix
from typing import Optional, Dict, Union, Any
import logging
import os
import hypernetx as hnx  # For hypergraph visualization (install via requirements.txt)

# Configure logger for visualization-specific debugging
logger = logging.getLogger("QuantumExperiment.Visualization")


class Visualizer:
    """
    Class for visualizing quantum experiment results, providing modular and extensible plotting tools.

    Attributes:
        None (static methods for now, but extensible for instance-specific behavior).
    """

    @staticmethod
    def plot_histogram(
        counts: Dict[str, int],
        state_type: Optional[str] = None,
        noise_type: Optional[str] = None,
        noise_enabled: Optional[bool] = None,
        save_path: Optional[str] = None,
        min_occurrences: int = 0,
    ) -> None:
        """
        Plots a histogram of quantum measurement results, with filtering and save options.

        Supports aggregation for large qubit counts and logging for hypergraph analysis.

        Args:
            counts (Dict[str, int]): Dictionary of measurement outcomes (state -> occurrences).
            state_type (str, optional): The quantum state type (e.g., "GHZ").
            noise_type (str, optional): The type of noise applied (e.g., "DEPOLARIZING").
            noise_enabled (bool, optional): Whether noise was enabled.
            save_path (str, optional): Path to save the plot (e.g., "histogram.png").
            min_occurrences (int): Minimum occurrences to include in plot (default 0, show all).

        Raises:
            ValueError: If counts is empty or invalid.
        """
        if counts is None:
            logger.warning("Counts object is None. No data to plot.")
            return

        # Convert Qiskit's Counts object to a dictionary if necessary
        try:
            counts = dict(counts)
        except TypeError:
            logger.error("Counts object could not be converted to a dictionary.")
            return

        # Ensure counts contains valid numeric data
        filtered_counts = {
            k: int(v)  # Convert values to integers
            for k, v in counts.items()
            if isinstance(v, (int, float)) and v >= min_occurrences
        }

        if not filtered_counts:
            logger.warning("No outcomes meet the minimum occurrences threshold.")
            return

        # Customize title based on experiment parameters
        title = f"{state_type or 'Quantum'} State Distribution"
        if noise_enabled:
            title += f" with {noise_type} Noise"
        logger.debug(
            f"Plotting histogram for {title}, filtering outcomes < {min_occurrences}"
        )

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.bar(
            filtered_counts.keys(),
            filtered_counts.values(),
            color="red" if noise_enabled else "blue",
            alpha=0.7,
        )
        plt.xlabel("Qubit State")
        plt.ylabel("Occurrences")
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Save or show plot
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved histogram to {save_path}")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_density_matrix(
        density_matrix: DensityMatrix,
        cmap: str = "viridis",
        show_real: bool = False,
        show_imag: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plots a heatmap of the density matrix, with real/imaginary options and save functionality.

        Args:
            density_matrix (DensityMatrix): The density matrix to plot.
            cmap (str, optional): The colormap to use (default "viridis").
            show_real (bool, optional): Show real part instead of absolute values (default False).
            show_imag (bool, optional): Show imaginary part instead of absolute values (default False).
            save_path (str, optional): Path to save the plot (e.g., "density.png").
        """
        if density_matrix is None or not isinstance(density_matrix, DensityMatrix):
            logger.warning("No valid density matrix available to plot.")
            return

        # Extract data for visualization
        dm_array = (
            np.real(density_matrix.data)
            if show_real
            else (
                np.imag(density_matrix.data)
                if show_imag
                else np.abs(density_matrix.data)
            )
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(dm_array, cmap=cmap, interpolation="nearest")
        plt.colorbar(label="Magnitude")
        plt.title("Density Matrix Heatmap")
        plt.xlabel("Basis State Index")
        plt.ylabel("Basis State Index")
        plt.grid(False)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved density matrix to {save_path}")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_hypergraph(
        correlation_data: Dict,
        state_type: Optional[str] = None,
        noise_type: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plots a hypergraph of quantum state correlations for research analysis.

        Args:
            correlation_data (Dict): Data on state correlations.
            state_type (str, optional): The quantum state type.
            noise_type (str, optional): The type of noise applied.
            save_path (str, optional): Path to save the plot.
        """
        if not correlation_data:
            logger.warning("No valid correlation data for hypergraph plotting.")
            return

        H = hnx.Hypergraph({state: {state} for state in correlation_data.keys()})

        plt.figure(figsize=(10, 6))
        hnx.drawing.draw(H)
        plt.title(f"{state_type or 'Quantum'} State Hypergraph")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved hypergraph to {save_path}")
            plt.close()
        else:
            plt.show()
