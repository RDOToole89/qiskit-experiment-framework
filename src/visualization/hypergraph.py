# src/visualization/hypergraph.py

import matplotlib.pyplot as plt
import hypernetx as hnx
import os
from typing import Optional, Dict
import logging

logger = logging.getLogger("QuantumExperiment.Visualization")

def plot_hypergraph(
    correlation_data: Dict,
    state_type: Optional[str] = None,
    noise_type: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots a hypergraph of quantum state correlations.

    Args:
        correlation_data (Dict): Data on state correlations.
        state_type (str, optional): Quantum state type.
        noise_type (str, optional): Noise applied.
        save_path (str, optional): File path to save the plot.
    """
    if not correlation_data:
        logger.warning("No valid correlation data for hypergraph plotting.")
        return

    # Here we build a simple hypergraph where each key forms a self-loop.
    H = hnx.Hypergraph({state: {state} for state in correlation_data.keys()})

    plt.figure(figsize=(10, 6))
    hnx.drawing.draw(H)
    plt.title(f"{state_type or 'Quantum'} State Hypergraph")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved hypergraph to {save_path}")
        plt.close()
    else:
        plt.show()
