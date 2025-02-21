# src/visualization/histogram.py

import matplotlib.pyplot as plt
import os
from typing import Optional, Dict
from src.utils.logger_setup import logger
from src.config.visualization_config import DEFAULT_BAR_COLOR, DEFAULT_FIGURE_SIZE
from src.config.file_config import DEFAULT_HISTOGRAM_SAVE_PATH


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
        counts (Dict[str, int]): Measurement outcomes.
        state_type (str, optional): Quantum state type.
        noise_type (str, optional): Noise model.
        noise_enabled (bool, optional): Whether noise is applied.
        save_path (str, optional): File path to save the plot.
        min_occurrences (int, optional): Minimum occurrences to show in the plot.
    """
    if not counts:
        logger.warning("No measurement data to plot.")
        return

    filtered_counts = {k: v for k, v in counts.items() if v >= min_occurrences}
    if not filtered_counts:
        logger.warning("No outcomes meet the minimum occurrences threshold.")
        return

    title = f"{state_type or 'Quantum'} State Distribution"
    if noise_enabled:
        title += f" with {noise_type} Noise"

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    plt.bar(
        filtered_counts.keys(),
        filtered_counts.values(),
        color=(
            DEFAULT_BAR_COLOR["with_noise"]
            if noise_enabled
            else DEFAULT_BAR_COLOR["no_noise"]
        ),
        alpha=0.7,
    )
    plt.xlabel("Qubit State")
    plt.ylabel("Occurrences")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    save_path = save_path or DEFAULT_HISTOGRAM_SAVE_PATH
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved histogram to {save_path}")
    plt.close()
