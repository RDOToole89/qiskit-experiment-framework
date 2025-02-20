# src/quantum_experiment/visualization.py

"""
Visualization functions for quantum experiments.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(counts, state_type=None, noise_type=None, noise_enabled=None):
    """
    Plots a histogram of quantum measurement results.

    Args:
        counts (dict): A dictionary of measurement outcomes.
        state_type (str, optional): The quantum state type (GHZ, W, etc.).
        noise_type (str, optional): The type of noise applied.
        noise_enabled (bool, optional): Whether noise was enabled.
    """

    # ✅ Ensure there is data to plot
    if not counts:
        print("⚠️ No data to plot.")
        return

    # ✅ Customize title based on experiment parameters
    title = f"{state_type or 'Quantum'} State Distribution"
    if noise_enabled:
        title += f" with {noise_type} Noise"

    # ✅ Plot histogram
    plt.figure(figsize=(8, 6))
    plt.bar(counts.keys(), counts.values(), color="red" if noise_enabled else "blue")
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_density_matrix(density_matrix, cmap="viridis"):
    """
    Plots a heatmap of the density matrix.

    Args:
        density_matrix (qiskit.quantum_info.DensityMatrix): The density matrix to plot.
        cmap (str, optional): The colormap to use for visualization (default: 'viridis').
    """

    # ✅ Ensure density matrix is valid
    if density_matrix is None:
        print("⚠️ No density matrix available to plot.")
        return

    # ✅ Extract absolute values for visualization
    dm_array = np.abs(density_matrix.data)

    # ✅ Plot density matrix as a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(dm_array, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Magnitude")
    plt.title("Density Matrix Heatmap (Absolute Values)")
    plt.xlabel("Basis State Index")
    plt.ylabel("Basis State Index")
    plt.xticks(range(dm_array.shape[0]))
    plt.yticks(range(dm_array.shape[1]))
    plt.grid(False)
    plt.show()
