# src/visualization/density_matrix.py

import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit.quantum_info import DensityMatrix
from typing import Optional
from src.utils.logger_setup import logger
from src.config.visualization_config import DEFAULT_CMAP, DEFAULT_FIGURE_SIZE
from src.config.file_config import DEFAULT_DENSITY_MATRIX_SAVE_PATH


def plot_density_matrix(
    density_matrix: DensityMatrix,
    cmap: str = DEFAULT_CMAP,
    show_real: bool = False,
    show_imag: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a heatmap of the density matrix.

    Args:
        density_matrix (DensityMatrix): The density matrix to plot.
        cmap (str, optional): Colormap for visualization.
        show_real (bool, optional): Show real part.
        show_imag (bool, optional): Show imaginary part.
        save_path (str, optional): Path to save the plot.
    """
    if not isinstance(density_matrix, DensityMatrix):
        logger.warning("No valid density matrix available to plot.")
        return

    dm_array = (
        np.real(density_matrix.data)
        if show_real
        else (
            np.imag(density_matrix.data) if show_imag else np.abs(density_matrix.data)
        )
    )

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.imshow(dm_array, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Magnitude")
    plt.title("Density Matrix Heatmap")
    plt.xlabel("Basis State Index")
    plt.ylabel("Basis State Index")
    plt.grid(False)

    save_path = save_path or DEFAULT_DENSITY_MATRIX_SAVE_PATH
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved density matrix to {save_path}")
    plt.close()
