# src/visualization/hypergraph.py

import matplotlib.pyplot as plt
import hypernetx as hnx  # For hypergraph visualization
import os
from typing import Dict, Optional
from src.utils.logger_setup import logger
from src.config.visualization_config import DEFAULT_FIGURE_SIZE
from src.config.file_config import DEFAULT_HYPERGRAPH_SAVE_PATH


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

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    hnx.drawing.draw(H)
    plt.title(f"{state_type or 'Quantum'} State Hypergraph")

    save_path = save_path or DEFAULT_HYPERGRAPH_SAVE_PATH
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    logger.info(f"Saved hypergraph to {save_path}")
    plt.close()
