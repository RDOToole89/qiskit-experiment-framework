# src/visualization/density_matrix.py

import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit.quantum_info import DensityMatrix
from typing import Optional
import logging

logger = logging.getLogger("QuantumExperiment.Visualization")

def plot_density_matrix(
    density_matrix: DensityMatrix,
    cmap: str = "viridis",
    show_real: bool = False,
    show_imag: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a heatmap of the density matrix.

    Args:
        density_matrix (DensityMatrix): Density matrix to plot.
        cmap (str, optional): Colormap for visualization.
        show_real (bool, optional): Show real part.
        show_imag (bool, optional): Show imaginary part.
        save_path (str, optional): File path to save the plot.
    """
    if density_matrix is None or not isinstance(density_matrix, DensityMatrix):
        logger.warning("No valid density matrix available to plot.")
        return

    dm_array = (
        np.real(density_matrix.data)
        if show_real
        else (np.imag(density_matrix.data) if show_imag else np.abs(density_matrix.data))
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
