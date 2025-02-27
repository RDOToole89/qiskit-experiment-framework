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
    state_type: Optional[str] = None,
    noise_type: Optional[str] = None,
) -> None:
    """
    Plots a heatmap of the density matrix with basis state labels.

    Args:
        density_matrix (DensityMatrix): Density matrix to plot.
        cmap (str, optional): Colormap for visualization.
        show_real (bool, optional): Show real part.
        show_imag (bool, optional): Show imaginary part.
        save_path (str, optional): File path to save the plot.
        state_type (str, optional): Quantum state type (e.g., GHZ, W, CLUSTER).
        noise_type (str, optional): Noise applied (e.g., DEPOLARIZING).
    """
    if density_matrix is None or not isinstance(density_matrix, DensityMatrix):
        logger.warning("No valid density matrix available to plot.")
        return

    # Extract the density matrix data as a numpy array
    dm_array = (
        np.real(density_matrix.data)
        if show_real
        else (np.imag(density_matrix.data) if show_imag else np.abs(density_matrix.data))
    )

    # Determine the number of qubits from the matrix size (2^n x 2^n)
    num_qubits = int(np.log2(dm_array.shape[0]))
    basis_states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
    basis_labels = [f"|{state}⟩" for state in basis_states]

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(dm_array, cmap=cmap, interpolation="nearest")

    # Add colorbar with appropriate label
    colorbar_label = (
        "Real Part" if show_real
        else "Imaginary Part" if show_imag
        else "Absolute Value"
    )
    plt.colorbar(im, label=colorbar_label)

    # Set title
    title = f"{state_type or 'Quantum'} State Density Matrix"
    if noise_type:
        title += f" with {noise_type} Noise"
    plt.title(title)

    # Set axis labels and ticks
    plt.xlabel("Basis State")
    plt.ylabel("Basis State")
    plt.xticks(ticks=range(len(basis_labels)), labels=basis_labels, rotation=45, ha="right")
    plt.yticks(ticks=range(len(basis_labels)), labels=basis_labels)

    # Add grid to separate basis states
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, color="gray")
    plt.minorticks_on()
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved density matrix plot to {save_path} (dimensions: {dm_array.shape})")
        plt.close()
    else:
        plt.show()
