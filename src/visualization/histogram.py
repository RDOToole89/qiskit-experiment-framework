# src/visualization/histogram.py

import matplotlib.pyplot as plt
import os
from typing import Optional, Dict
import logging

logger = logging.getLogger("QuantumExperiment.Visualization")

def plot_histogram(
    counts: Dict[str, int],
    state_type: Optional[str] = None,
    noise_type: Optional[str] = None,
    noise_enabled: Optional[bool] = None,
    save_path: Optional[str] = None,
    min_occurrences: int = 0,
) -> None:
    """
    Plots a histogram of quantum measurement results.

    Args:
        counts (Dict[str, int]): Dictionary of measurement outcomes.
        state_type (str, optional): Quantum state type.
        noise_type (str, optional): Noise model used.
        noise_enabled (bool, optional): Whether noise was applied.
        save_path (str, optional): File path to save the plot.
        min_occurrences (int): Minimum occurrences to display.
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

    filtered_counts = {
        k: int(v)
        for k, v in counts.items()
        if isinstance(v, (int, float)) and v >= min_occurrences
    }

    if not filtered_counts:
        logger.warning("No outcomes meet the minimum occurrences threshold.")
        return

    title = f"{state_type or 'Quantum'} State Distribution"
    if noise_enabled:
        title += f" with {noise_type} Noise"

    logger.debug(f"Plotting histogram: {title}")

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

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved histogram to {save_path}")
        plt.close()
    else:
        plt.show()
