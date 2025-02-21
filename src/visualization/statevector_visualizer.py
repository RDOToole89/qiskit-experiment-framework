import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit.quantum_info import Statevector
import logging

# Configure logger for visualization
logger = logging.getLogger("QuantumExperiment.StatevectorVisualization")


def plot_statevector(statevector_file: str, save_path: str = None) -> None:
    """
    Plots the probability amplitudes of a saved statevector and optionally saves it to a file.

    Args:
        statevector_file (str): Path to the saved .npy statevector file.
        save_path (str, optional): Path to save the plot (e.g., "results/statevector/plot.png").

    Raises:
        FileNotFoundError: If the statevector file does not exist.
    """
    if not os.path.exists(statevector_file):
        logger.error(f"Statevector file not found: {statevector_file}")
        raise FileNotFoundError(f"Statevector file not found: {statevector_file}")

    # Load statevector from file
    statevector_data = np.load(statevector_file)
    statevector = Statevector(statevector_data)

    # Compute probabilities
    probabilities = np.abs(statevector.data) ** 2
    labels = [
        bin(i)[2:].zfill(len(statevector.dims())) for i in range(len(probabilities))
    ]

    # Plot statevector amplitudes
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probabilities, color="blue", alpha=0.7)
    plt.xlabel("Basis States")
    plt.ylabel("Probability Amplitude")
    plt.title("Quantum Statevector Probabilities")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved statevector plot to {save_path}")
        plt.close()
    else:
        plt.show()
        logger.info(f"Displayed statevector plot for {statevector_file}")


import matplotlib.pyplot as plt
import numpy as np
import os
from qiskit.quantum_info import Statevector
import logging

# Configure logger for visualization
logger = logging.getLogger("QuantumExperiment.StatevectorVisualization")


def plot_statevector(statevector_file: str, save_path: str = None) -> None:
    """
    Plots the probability amplitudes of a saved statevector and optionally saves it to a file.

    Args:
        statevector_file (str): Path to the saved .npy statevector file.
        save_path (str, optional): Path to save the plot (e.g., "results/statevector/plot.png").

    Raises:
        FileNotFoundError: If the statevector file does not exist.
    """
    if not os.path.exists(statevector_file):
        logger.error(f"Statevector file not found: {statevector_file}")
        raise FileNotFoundError(f"Statevector file not found: {statevector_file}")

    # Load statevector from file
    statevector_data = np.load(statevector_file)
    statevector = Statevector(statevector_data)

    # Compute probabilities
    probabilities = np.abs(statevector.data) ** 2
    labels = [
        bin(i)[2:].zfill(len(statevector.dims())) for i in range(len(probabilities))
    ]

    # Plot statevector amplitudes
    plt.figure(figsize=(10, 6))
    plt.bar(labels, probabilities, color="blue", alpha=0.7)
    plt.xlabel("Basis States")
    plt.ylabel("Probability Amplitude")
    plt.title("Quantum Statevector Probabilities")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved statevector plot to {save_path}")
        plt.close()
    else:
        plt.show()
        logger.info(f"Displayed statevector plot for {statevector_file}")
