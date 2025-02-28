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

# Gell-Mann matrices for SU(3) symmetry analysis
GELL_MANN = [
    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # λ1
    np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]),  # λ2
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),  # λ3
    np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),  # λ4
    np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]),  # λ5
    np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),  # λ6
    np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]),  # λ7
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]]) / np.sqrt(3),  # λ8
]


def compute_su2_symmetry(counts: Dict, num_qubits: int, shots: float) -> Dict:
    """
    Computes SU(2) symmetry metrics using Pauli correlations (XX, YY, ZZ) for pairwise qubits.

    Args:
        counts (Dict): Measurement counts.
        num_qubits (int): Number of qubits.
        shots (float): Total number of shots.

    Returns:
        Dict: Pauli correlations for each pair of qubits and SU(2) symmetry metric.
    """
    correlations = {"XX": {}, "YY": {}, "ZZ": {}}
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            zz_corr = 0.0
            for bitstring, count in counts.items():
                bit_i, bit_j = int(bitstring[i]), int(bitstring[j])
                zz_value = (-1) ** (bit_i + bit_j)
                zz_corr += zz_value * (count / shots)
            correlations["ZZ"][(i, j)] = zz_corr
    zz_values = list(correlations["ZZ"].values())
    su2_symmetry = np.var(zz_values) if zz_values else 0.0
    return {"correlations": correlations, "su2_symmetry": su2_symmetry}


def compute_su3_symmetry(density_matrix: np.ndarray, num_qubits: int) -> float:
    """
    Computes SU(3) symmetry metrics using Gell-Mann matrices for 3-qubit subsets.

    Args:
        density_matrix (np.ndarray): Density matrix.
        num_qubits (int): Number of qubits.

    Returns:
        float: SU(3) symmetry metric (variance of Gell-Mann expectations).
    """
    if num_qubits < 3:
        return 0.0
    rho_123 = partial_trace(density_matrix, keep=[0, 1, 2], dims=[2] * num_qubits)
    expectations = []
    for gm in GELL_MANN:
        expectation = np.trace(rho_123 @ gm).real
        expectations.append(expectation)
    return np.var(expectations)


def compute_bloch_vector(rho: np.ndarray) -> tuple:
    """
    Computes the Bloch vector (x, y, z) for a single-qubit density matrix.

    Args:
        rho (np.ndarray): Single-qubit density matrix.

    Returns:
        tuple: (x, y, z) coordinates on the Bloch sphere.
    """
    pauli_x = Pauli("X").to_matrix()
    pauli_y = Pauli("Y").to_matrix()
    pauli_z = Pauli("Z").to_matrix()
    x = np.trace(rho @ pauli_x).real
    y = np.trace(rho @ pauli_y).real
    z = np.trace(rho @ pauli_z).real
    return (x, y, z)


def plot_bloch_sphere_vectors(
    bloch_vectors: List[Dict[int, tuple]],
    time_steps: List[float],
    save_path: str,
) -> None:
    """
    Plots the Bloch sphere trajectories for each qubit over time.

    Args:
        bloch_vectors (List[Dict[int, tuple]]): List of Bloch vectors per qubit at each timestep.
        time_steps (List[float]): Timesteps.
        save_path (str): Base path to save the plots.
    """
    from mpl_toolkits.mplot3d import Axes3D

    num_qubits = len(bloch_vectors[0])
    for qubit in range(num_qubits):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"Bloch Sphere Trajectory - Qubit {qubit}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)
        xs = [bv[qubit][0] for bv in bloch_vectors]
        ys = [bv[qubit][1] for bv in bloch_vectors]
        zs = [bv[qubit][2] for bv in bloch_vectors]
        ax.plot(xs, ys, zs, marker="o", label=f"Qubit {qubit}")
        for i in range(len(xs) - 1):
            ax.quiver(
                xs[i],
                ys[i],
                zs[i],
                xs[i + 1] - xs[i],
                ys[i + 1] - ys[i],
                zs[i + 1] - zs[i],
                color="blue",
                alpha=0.5,
                arrow_length_ratio=0.1,
            )
        ax.legend()
        plt.savefig(
            f"{save_path}_bloch_qubit_{qubit}.png", bbox_inches="tight", dpi=300
        )
        logger.info(f"Saved Bloch sphere plot to {save_path}_bloch_qubit_{qubit}.png")
        plt.close()


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

    Returns:
        Dict: Dictionary of edges with correlation weights.
    """
    edges = {}
    edge_id = 0
    shots = sum(correlation_data.values()) if mode == "qasm" else 1
    threshold = config.get("threshold", 0.1 if mode == "qasm" else 0.01)
    max_order = config.get("max_order", 2)

    if mode == "qasm":
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
        state_type (str, optional): Quantum state type.
        noise_type (str, optional): Noise applied.
        save_path (str, optional): File path to save the plot.
        time_steps (List[float], optional): Timesteps for dynamic visualization.
        config (Dict, optional): Configuration dictionary.
    """
    config = config or {}
    config.setdefault("max_order", 2)
    config.setdefault("threshold", None)
    config.setdefault("symmetry_analysis", False)
    config.setdefault("plot_transitions", False)
    config.setdefault("plot_bloch", False)
    config.setdefault("node_color", "blue")
    config.setdefault("edge_color", "red")

    if time_steps is not None and isinstance(correlation_data, list):
        if config["plot_bloch"]:
            bloch_vectors = []
            for data in correlation_data:
                if "density" in data:
                    density_matrix = np.array(data["density"])
                    num_qubits = int(np.log2(density_matrix.shape[0]))
                    qubit_bloch = {}
                    for qubit in range(num_qubits):
                        rho_qubit = partial_trace(
                            density_matrix, keep=[qubit], dims=[2] * num_qubits
                        )
                        bloch_vector = compute_bloch_vector(rho_qubit)
                        qubit_bloch[qubit] = bloch_vector
                    bloch_vectors.append(qubit_bloch)
            plot_bloch_sphere_vectors(bloch_vectors, time_steps, save_path)
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

    mode = "density" if "density" in correlation_data else "qasm"
    if mode == "density":
        density_matrix = np.array(correlation_data["density"])
        num_qubits = int(np.log2(density_matrix.shape[0]))
    else:
        num_qubits = len(next(iter(correlation_data.keys())))
        shots = sum(correlation_data.values())

    nodes = [f"q{i}" for i in range(num_qubits)]
    edges = compute_correlations(correlation_data, num_qubits, mode, config)
    if not edges:
        logger.warning("No significant correlations found for hypergraph plotting.")
        return

    hyperedges = {name: edge_nodes for name, (edge_nodes, _) in edges.items()}
    H = hnx.Hypergraph(hyperedges)

    plt.figure(figsize=(10, 6))
    # Use the corrected keyword 'nodes_kwargs' (with an 's') below.
    hnx.drawing.draw(
        H,
        node_labels={node: node for node in nodes},
        edge_labels={
            name: f"{props['weight']:.2f}"
            for name, (edge_nodes, props) in edges.items()
        },
        nodes_kwargs={"color": config["node_color"]},
        edges_kwargs={"color": config["edge_color"]},
    )
    title = f"{state_type or 'Quantum'} State Hypergraph"
    if noise_type:
        title += f" with {noise_type} Noise"
    if time_step is not None:
        title += f" (t={time_step:.2f})"
    plt.title(title)

    if config.get("symmetry_analysis"):
        if mode == "qasm":
            parity = compute_parity_distribution(correlation_data, num_qubits)
            perm_sym = compute_permutation_symmetric_correlations(
                correlation_data, num_qubits, shots
            )
            su2_sym = compute_su2_symmetry(correlation_data, num_qubits, shots)
            symmetry_text = (
                f"Parity (Even/Odd): {parity['even']:.2f}/{parity['odd']:.2f}\n"
                f"Perm. Sym. ZZ: {perm_sym:.2f}\n"
                f"SU(2) Symmetry (var): {su2_sym['su2_symmetry']:.2f}"
            )
            plt.text(
                0.05,
                0.95,
                symmetry_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )
        else:
            conditional_corrs = compute_conditional_correlations(
                density_matrix, num_qubits
            )
            avg_corr = np.mean(list(conditional_corrs.values()))
            su3_sym = compute_su3_symmetry(density_matrix, num_qubits)
            symmetry_text = (
                f"Avg. Conditional Corr.: {avg_corr:.2f}\n"
                f"SU(3) Symmetry (var): {su3_sym:.2f}"
            )
            plt.text(
                0.05,
                0.95,
                symmetry_text,
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
                    if transition_prob > 0.01:
                        G.add_edge(
                            state, next_state, weight=transition_prob, t=time_steps[t]
                        )
    pos = nx.spring_layout(G)
    for t in time_steps[:-1]:
        plt.figure(figsize=(10, 6))
        edges = [(u, v) for u, v, d in G.edges(data=True) if d["t"] == t]
        if not edges:
            plt.close()
            continue
        weights = [G[u][v]["weight"] * 5 for u, v in edges]
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
