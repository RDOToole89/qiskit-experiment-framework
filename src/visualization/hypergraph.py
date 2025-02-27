# src/visualization/hypergraph.py

import matplotlib.pyplot as plt
import hypernetx as hnx
import numpy as np
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
        correlation_data (Dict): Data on state correlations (e.g., counts for QASM, density matrix for density mode).
        state_type (str, optional): Quantum state type (e.g., GHZ, W, CLUSTER).
        noise_type (str, optional): Noise applied (e.g., DEPOLARIZING).
        save_path (str, optional): File path to save the plot.
    """
    if not correlation_data:
        logger.warning("No valid correlation data for hypergraph plotting.")
        return

    # Initialize the hypergraph edges dictionary
    edges = {}

    # Determine if this is density mode (correlation_data has "density" key)
    if "density" in correlation_data:
        # Extract the density matrix (numpy array)
        density_matrix = np.array(correlation_data["density"])
        num_qubits = int(np.log2(density_matrix.shape[0]))  # Infer number of qubits from matrix size

        # Create nodes for each qubit
        nodes = [f"q{i}" for i in range(num_qubits)]

        # Compute correlations from the density matrix
        # We'll use the absolute values of off-diagonal elements to represent correlation strength
        threshold = 0.01  # Correlation threshold to avoid cluttering with small values
        edge_id = 0
        for i in range(density_matrix.shape[0]):
            for j in range(i + 1, density_matrix.shape[1]):
                corr = abs(density_matrix[i, j])
                if corr > threshold:
                    # Map basis states to qubits based on bitstrings
                    bitstring_i = format(i, f'0{num_qubits}b')
                    bitstring_j = format(j, f'0{num_qubits}b')
                    # Find the qubits that differ between the two bitstrings
                    differing_qubits = [k for k in range(num_qubits) if bitstring_i[k] != bitstring_j[k]]
                    if len(differing_qubits) >= 2:  # Only add hyperedges for 2+ qubits
                        # Create a hyperedge connecting the differing qubits
                        edge_nodes = frozenset([f"q{k}" for k in differing_qubits])
                        edges[f"e{edge_id}"] = (edge_nodes, {"weight": corr})
                        edge_id += 1

    else:
        # QASM mode: correlation_data contains counts
        counts = correlation_data
        num_qubits = len(next(iter(counts.keys())))  # Infer number of qubits from bitstring length

        # Create nodes for each qubit
        nodes = [f"q{i}" for i in range(num_qubits)]

        # Compute pairwise correlations (e.g., <Z_i Z_j> expectations)
        shots = sum(counts.values())
        edge_id = 0
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                zz_corr = 0.0
                for bitstring, count in counts.items():
                    # Compute <Z_i Z_j> = (-1)^(bit_i + bit_j)
                    bit_i = int(bitstring[i])
                    bit_j = int(bitstring[j])
                    zz_value = (-1) ** (bit_i + bit_j)
                    zz_corr += zz_value * (count / shots)
                if abs(zz_corr) > 0.1:  # Threshold to avoid clutter
                    edge_nodes = frozenset([f"q{i}", f"q{j}"])
                    edges[f"e{edge_id}"] = (edge_nodes, {"weight": zz_corr})
                    edge_id += 1

    # Create the hypergraph
    if not edges:
        logger.warning("No significant correlations found for hypergraph plotting.")
        return

    # Create the hypergraph with nodes and edges
    hyperedges = {name: edge_nodes for name, (edge_nodes, _) in edges.items()}
    H = hnx.Hypergraph(hyperedges)

    # Plot the hypergraph
    plt.figure(figsize=(10, 6))
    hnx.drawing.draw(
        H,
        node_labels={node: node for node in nodes},
        edge_labels={name: f"{props['weight']:.2f}" for name, (edge_nodes, props) in edges.items()}
    )
    title = f"{state_type or 'Quantum'} State Hypergraph"
    if noise_type:
        title += f" with {noise_type} Noise"
    plt.title(title)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved hypergraph to {save_path}")
        plt.close()
    else:
        plt.show()
