# src/visualization/hypergraph.py

import matplotlib.pyplot as plt
import hypernetx as hnx
import numpy as np
import networkx as nx
import os
from typing import Optional, Dict, List, Union
import logging
from qiskit.quantum_info import partial_trace, Pauli
from itertools import combinations

logger = logging.getLogger("QuantumExperiment.Visualization")


def compute_correlations(
    correlation_data: Dict,
    num_qubits: int,
    mode: str,
    config: Dict,
) -> Dict:
    """
    Computes correlations from correlation data based on configuration.

    Args:
        correlation_data (Dict): Counts or density matrix data.
        num_qubits (int): Number of qubits.
        mode (str): 'qasm' or 'density'.
        config (Dict): Configuration for correlation computation.
            - max_order (int): Maximum order of correlations (e.g., 2 for pairwise, 3 for 3-qubit).
            - threshold (float): Correlation threshold for including edges.

    Returns:
        Dict: Dictionary of edges with correlation weights.
    """
    edges = {}
    edge_id = 0
    shots = sum(correlation_data.values()) if mode == "qasm" else 1
    threshold = config.get("threshold", 0.1 if mode == "qasm" else 0.01)
    max_order = config.get("max_order", 2)  # Default to pairwise correlations

    if mode == "qasm":
        # Compute higher-order correlations up to max_order
        for r in range(2, max_order + 1):
            for qubit_subset in combinations(range(num_qubits), r):
                corr = 0.0
                for bitstring, count in correlation_data.items():
                    bits = [int(bitstring[i]) for i in qubit_subset]
                    value = (-1) ** sum(bits)
                    corr += value * (count / shots)
                if abs(corr) > threshold:
                    edge_nodes = frozenset([f"q{i}" for i in qubit_subset])
                    edges[f"e{edge_id}"] = (edge_nodes, {"weight": corr})
                    edge_id += 1
    else:
        # Density mode
        density_matrix = np.array(correlation_data["density"])
        for i in range(density_matrix.shape[0]):
            for j in range(i + 1, density_matrix.shape[1]):
                corr = abs(density_matrix[i, j])
                if corr > threshold:
                    bitstring_i = format(i, f"0{num_qubits}b")
                    bitstring_j = format(j, f"0{num_qubits}b")
                    differing_qubits = [
                        k for k in range(num_qubits) if bitstring_i[k] != bitstring_j[k]
                    ]
                    if len(differing_qubits) >= 2:
                        edge_nodes = frozenset([f"q{k}" for k in differing_qubits])
                        edges[f"e{edge_id}"] = (edge_nodes, {"weight": corr})
                        edge_id += 1
    return edges


def plot_hypergraph(
    correlation_data: Union[Dict, List[Dict]],
    state_type: Optional[str] = None,
    noise_type: Optional[str] = None,
    save_path: Optional[str] = None,
    time_steps: Optional[List[float]] = None,
    config: Optional[Dict] = None,
) -> None:
    """
    Plots a hypergraph of quantum state correlations with configurable options.

    Args:
        correlation_data (Union[Dict, List[Dict]]): Counts or density matrix data, or list for time evolution.
        state_type (str, optional): Quantum state type (e.g., GHZ, W, CLUSTER).
        noise_type (str, optional): Noise applied (e.g., DEPOLARIZING).
        save_path (str, optional): File path to save the plot.
        time_steps (List[float], optional): Timesteps for dynamic visualization.
        config (Dict, optional): Configuration dictionary.
            - max_order (int): Maximum order of correlations (default: 2).
            - threshold (float): Correlation threshold (default: 0.1 for QASM, 0.01 for density).
            - symmetry_analysis (bool): Compute and display symmetry metrics (default: False).
            - plot_transitions (bool): Plot error transition graph (default: False).
    """
    config = config or {}
    config.setdefault("max_order", 2)
    config.setdefault("threshold", None)  # Will be set based on mode
    config.setdefault("symmetry_analysis", False)
    config.setdefault("plot_transitions", False)

    if time_steps is not None and isinstance(correlation_data, list):
        for step, data in enumerate(correlation_data):
            plot_single_hypergraph(
                data,
                state_type,
                noise_type,
                f"{save_path}_step_{step}.png" if save_path else None,
                time_steps[step],
                config,
            )
        if config.get("plot_transitions"):
            plot_error_transition_graph(correlation_data, time_steps, save_path)
    else:
        plot_single_hypergraph(
            correlation_data,
            state_type,
            noise_type,
            save_path,
            None,
            config,
        )


def plot_single_hypergraph(
    correlation_data: Dict,
    state_type: Optional[str],
    noise_type: Optional[str],
    save_path: Optional[str],
    time_step: Optional[float],
    config: Dict,
) -> None:
    """
    Plots a single hypergraph for given correlation data.
    """
    if not correlation_data:
        logger.warning("No valid correlation data for hypergraph plotting.")
        return

    # Determine mode and number of qubits
    mode = "density" if "density" in correlation_data else "qasm"
    if mode == "density":
        density_matrix = np.array(correlation_data["density"])
        num_qubits = int(np.log2(density_matrix.shape[0]))
    else:
        num_qubits = len(next(iter(correlation_data.keys())))
        shots = sum(correlation_data.values())

    # Create nodes for each qubit
    nodes = [f"q{i}" for i in range(num_qubits)]

    # Compute correlations
    edges = compute_correlations(correlation_data, num_qubits, mode, config)

    if not edges:
        logger.warning("No significant correlations found for hypergraph plotting.")
        return

    # Create the hypergraph
    hyperedges = {name: edge_nodes for name, (edge_nodes, _) in edges.items()}
    H = hnx.Hypergraph(hyperedges)

    # Plot the hypergraph
    plt.figure(figsize=(10, 6))
    hnx.drawing.draw(
        H,
        node_labels={node: node for node in nodes},
        edge_labels={
            name: f"{props['weight']:.2f}"
            for name, (edge_nodes, props) in edges.items()
        },
    )
    title = f"{state_type or 'Quantum'} State Hypergraph"
    if noise_type:
        title += f" with {noise_type} Noise"
    if time_step is not None:
        title += f" (t={time_step:.2f})"
    plt.title(title)

    if config.get("symmetry_analysis"):
        # Compute and display symmetry metrics
        if mode == "qasm":
            parity = compute_parity_distribution(correlation_data, num_qubits)
            perm_sym = compute_permutation_symmetric_correlations(
                correlation_data, num_qubits, shots
            )
            plt.text(
                0.05,
                0.95,
                f"Parity (Even/Odd): {parity['even']:.2f}/{parity['odd']:.2f}\nPerm. Sym. ZZ: {perm_sym:.2f}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )
        else:
            conditional_corrs = compute_conditional_correlations(
                density_matrix, num_qubits
            )
            avg_corr = np.mean([corr for corr in conditional_corrs.values()])
            plt.text(
                0.05,
                0.95,
                f"Avg. Conditional Corr.: {avg_corr:.2f}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved hypergraph to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_error_transition_graph(
    counts_list: List[Dict], time_steps: List[float], save_path: str
) -> None:
    """
    Plots a transition graph of error propagation over time.

    Args:
        counts_list (List[Dict]): List of counts dictionaries at each timestep.
        time_steps (List[float]): Timesteps corresponding to counts.
        save_path (str): Base path to save the plots.
    """
    num_qubits = len(next(iter(counts_list[0].keys())))
    G = nx.DiGraph()
    states = [format(i, f"0{num_qubits}b") for i in range(2**num_qubits)]
    for state in states:
        G.add_node(state)
    shots = sum(counts_list[0].values())
    for t in range(len(counts_list) - 1):
        counts_t = counts_list[t]
        counts_t1 = counts_list[t + 1]
        for state in states:
            prob_t = counts_t.get(state, 0) / shots
            for next_state in states:
                prob_t1 = counts_t1.get(next_state, 0) / shots
                if prob_t > 0 and prob_t1 > 0 and state != next_state:
                    transition_prob = prob_t1 / prob_t
                    if transition_prob > 0.01:  # Threshold to avoid clutter
                        G.add_edge(
                            state, next_state, weight=transition_prob, t=time_steps[t]
                        )

    # Plot the graph for each timestep
    pos = nx.spring_layout(G)
    for t in time_steps[:-1]:
        plt.figure(figsize=(10, 6))
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["t"] == t]
        if not edges:
            plt.close()
            continue
        weights = [
            G[u][v]["weight"] * 5 for u, v in edges
        ]  # Scale weights for visibility
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
        nx.draw_networkx_labels(G, pos)
        plt.title(f"Error Transitions at t={t:.2f}")
        plt.savefig(
            f"{save_path}_transition_t{t:.2f}.png", bbox_inches="tight", dpi=300
        )
        logger.info(f"Saved transition graph to {save_path}_transition_t{t:.2f}.png")
        plt.close()


def compute_parity_distribution(counts: Dict, num_qubits: int) -> Dict:
    parity_counts = {"even": 0, "odd": 0}
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        parity = sum(int(bit) for bit in bitstring) % 2
        parity_counts["even" if parity == 0 else "odd"] += count / shots
    return parity_counts


def compute_permutation_symmetric_correlations(
    counts: Dict, num_qubits: int, shots: float
) -> float:
    zz_symmetric = 0.0
    pairs = 0
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            zz_corr = 0.0
            for bitstring, count in counts.items():
                bit_i, bit_j = int(bitstring[i]), int(bitstring[j])
                zz_value = (-1) ** (bit_i + bit_j)
                zz_corr += zz_value * (count / shots)
            zz_symmetric += zz_corr
            pairs += 1
    return zz_symmetric / pairs if pairs > 0 else 0.0


def compute_conditional_correlations(
    density_matrix: np.ndarray, num_qubits: int
) -> Dict:
    conditional_corrs = {}
    pauli_z = Pauli("Z").to_matrix()
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                rho_ij = partial_trace(
                    density_matrix, keep=[i, j], dims=[2] * num_qubits
                )
                zz_corr = np.trace(np.kron(pauli_z, pauli_z) @ rho_ij)
                conditional_corrs[(i, j)] = zz_corr.real
    return conditional_corrs
