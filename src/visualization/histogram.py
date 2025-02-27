# src/visualization/histogram.py

import matplotlib.pyplot as plt
import os
from typing import Optional, Dict
import logging
from natsort import natsorted  # For natural sorting of basis states

logger = logging.getLogger("QuantumExperiment.Visualization")

def plot_histogram(
    counts: Dict[str, int],
    state_type: Optional[str] = None,
    noise_type: Optional[str] = None,
    noise_enabled: Optional[bool] = None,
    save_path: Optional[str] = None,
    min_occurrences: int = 0,
    num_qubits: Optional[int] = None,  # Add num_qubits parameter for context
) -> None:
    """
    Plots a histogram of quantum measurement results.

    Args:
        counts (Dict[str, int]): Dictionary of measurement outcomes.
        state_type (str, optional): Quantum state type (e.g., GHZ, W, CLUSTER).
        noise_type (str, optional): Noise model used (e.g., DEPOLARIZING).
        noise_enabled (bool, optional): Whether noise was applied.
        save_path (str, optional): File path to save the plot.
        min_occurrences (int): Minimum occurrences to display.
        num_qubits (int, optional): Number of qubits in the system.
    """
    if counts is None:
        logger.warning("Counts object is None. No data to plot.")
        return

    # Ensure counts is a dictionary of numeric values
    try:
        counts = dict(counts)
    except TypeError:
        logger.error("Counts object could not be converted to a dictionary.")
        return

    # Filter counts based on min_occurrences
    filtered_counts = {
        k: int(v)
        for k, v in counts.items()
        if isinstance(v, (int, float)) and v >= min_occurrences
    }

    if not filtered_counts:
        logger.warning("No outcomes meet the minimum occurrences threshold.")
        return

    # Sort the basis states in natural order (e.g., 000, 001, ..., 111)
    states = natsorted(filtered_counts.keys())
    occurrences = [filtered_counts[state] for state in states]

    # Compute total shots for probability calculation
    total_shots = sum(occurrences)
    probabilities = [count / total_shots for count in occurrences]

    # Determine number of qubits if not provided
    if num_qubits is None:
        num_qubits = len(states[0]) if states else 1

    # Create the histogram
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        states,
        probabilities,
        color="red" if noise_enabled else "blue",
        alpha=1.0,  # Fully opaque bars
    )

    # Add counts as labels above the bars
    for bar, count in zip(bars, occurrences):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Set labels and title
    plt.xlabel("Basis State")
    plt.ylabel("Probability")
    plt.title(
        f"{state_type or 'Quantum'} State Distribution ({num_qubits} qubits, {total_shots} shots)\n"
        f"{'with ' + noise_type + ' Noise' if noise_enabled else 'No Noise'}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save or display the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(
            f"Saved histogram to {save_path} (states: {len(states)}, total shots: {total_shots})"
        )
        plt.close()
    else:
        plt.show()
