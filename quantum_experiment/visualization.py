import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(counts, state_type, noise_type, noise_enabled):
    color = "red" if noise_enabled else "blue"
    title = f"{state_type} State Distribution {'With ' + noise_type + ' Noise' if noise_enabled else 'Without Noise'}"

    plt.bar(counts.keys(), counts.values(), color=color)
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.show()


def plot_density_matrix(density_matrix):
    dm_array = np.abs(density_matrix.data)
    plt.figure(figsize=(8, 6))
    plt.imshow(dm_array, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Magnitude")
    plt.title("Density Matrix Heatmap (Absolute Values)")
    plt.xlabel("Basis State Index")
    plt.ylabel("Basis State Index")
    plt.show()
