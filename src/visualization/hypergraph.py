# src/visualization/hypergraph.py

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import hypernetx as hnx
from typing import Optional, Dict, List, Union
from itertools import combinations
from qiskit.quantum_info import partial_trace, Pauli
from scipy.spatial import ConvexHull


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
    if num_qubits < 3:
        return 0.0
    rho_123 = partial_trace(density_matrix, keep=[0, 1, 2], dims=[2] * num_qubits)
    expectations = []
    for gm in GELL_MANN:
        expectation = np.trace(rho_123 @ gm).real
        expectations.append(expectation)
    return np.var(expectations)


def compute_bloch_vector(rho: np.ndarray) -> tuple:
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
    Plots a hypergraph of quantum state correlations with enhanced scientific visualization.
    """
    config = config or {}
    config.setdefault("max_order", 2)
    config.setdefault("threshold", None)
    config.setdefault("symmetry_analysis", False)
    config.setdefault("plot_transitions", False)
    config.setdefault("plot_bloch", False)
    config.setdefault("node_color", "blue")
    config.setdefault("edge_color", "red")  # fallback color

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
            correlation_data, state_type, noise_type, save_path, None, config
        )


def plot_single_hypergraph(
    correlation_data: dict,
    state_type: str,
    noise_type: str,
    save_path: str,
    time_step: float,
    config: dict,
) -> None:
    """
    Plots a hypergraph of correlations, plus an analysis box below the plot.
    """
    if not correlation_data:
        print("No valid correlation data for hypergraph plotting.")
        return

    # Distinguish QASM vs. density mode
    mode = "density" if "density" in correlation_data else "qasm"
    if mode == "density":
        density_matrix = np.array(correlation_data["density"])
        num_qubits = int(np.log2(density_matrix.shape[0]))
    else:
        first_key = next(iter(correlation_data.keys()))
        num_qubits = len(first_key)
        shots = sum(correlation_data.values())

    # Build edges from correlation
    edges = compute_correlations(correlation_data, num_qubits, mode, config)
    if not edges:
        print("No significant correlations found for hypergraph plotting.")
        return

    # Collect correlation values for color scaling and stats
    all_corrs = [props["weight"] for (_, props) in edges.values()]
    min_corr_val = min(all_corrs)
    max_corr_val = max(all_corrs)
    mean_corr_val = np.mean(all_corrs)
    abs_max_corr = max(abs(c) for c in all_corrs)

    # Set up figure with two subplots:
    #  - top for the hypergraph
    #  - bottom for the analysis text
    fig = plt.figure(figsize=(10, 8))
    # heights: 4 for the graph, 1 for the text
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_analysis = fig.add_subplot(gs[1, 0])
    ax_analysis.set_axis_off()  # We'll just place text here

    # Create a Hypernetx hypergraph
    Hedges = {}
    for edge_key, (edge_nodes, _) in edges.items():
        # use the frozenset of node labels as the key
        Hedges[frozenset(edge_nodes)] = frozenset(edge_nodes)
    H = hnx.Hypergraph(Hedges)

    # Position nodes with a spring layout
    pos = nx.spring_layout(H, seed=42)

    # Build style info for each edge
    cmap = cm.RdYlGn
    norm = mcolors.Normalize(vmin=-abs_max_corr, vmax=abs_max_corr)
    scale_factor = 4.0

    edge_styles = {}
    edge_labels = {}

    for edge_key, (edge_nodes, props) in edges.items():
        corr_val = props["weight"]
        color = cmap(norm(corr_val))
        linewidth = 1 + scale_factor * (abs(corr_val) / abs_max_corr)
        edge_styles[frozenset(edge_nodes)] = {
            "color": color,
            "linewidth": linewidth,
        }

        label_text = f"{corr_val:.2f}"
        if abs(corr_val) == abs_max_corr:
            label_text += " *"  # highlight max
        edge_labels[frozenset(edge_nodes)] = label_text

    # Draw the graph on ax_graph
    # Draw nodes
    nx.draw_networkx_nodes(
        H, pos, node_color=config.get("node_color", "blue"), ax=ax_graph
    )
    nx.draw_networkx_labels(H, pos, ax=ax_graph)

    # For each hyperedge, we draw a polygon
    for ekey, style_dict in edge_styles.items():
        pts = np.array([pos[node] for node in ekey])
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
            except:
                poly = pts
        else:
            poly = pts

        patch = plt.Polygon(
            poly,
            closed=True,
            fill=False,
            edgecolor=style_dict["color"],
            linewidth=style_dict["linewidth"],
        )
        ax_graph.add_patch(patch)

        # place label at centroid
        centroid = poly.mean(axis=0)
        ax_graph.text(
            centroid[0],
            centroid[1],
            edge_labels[ekey],
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )

    # Title
    title_str = f"{state_type or 'Quantum'} State Hypergraph"
    if noise_type:
        title_str += f" with {noise_type} Noise"
    if time_step is not None:
        title_str += f" (t={time_step:.2f})"
    ax_graph.set_title(title_str)

    # Add colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(
        sm, ax=ax_graph, orientation="vertical", label="Correlation Value"
    )

    # --- Build the analysis text that always shows some basic info ---
    analysis_lines = []
    analysis_lines.append(r"**Basic Correlation Stats**:")
    if mode == "qasm":
        analysis_lines.append(f"- Shots Used: {shots}")
    analysis_lines.append(f"- Min Corr: {min_corr_val:.2f}")
    analysis_lines.append(f"- Max Corr: {max_corr_val:.2f}")
    analysis_lines.append(f"- Mean Corr: {mean_corr_val:.2f}")

    # If symmetry analysis was chosen, add more info
    if config.get("symmetry_analysis"):
        if mode == "qasm":
            from . import (
                compute_parity_distribution,
                compute_permutation_symmetric_correlations,
                compute_su2_symmetry,
            )

            parity = compute_parity_distribution(correlation_data, num_qubits)
            perm_sym = compute_permutation_symmetric_correlations(
                correlation_data, num_qubits, shots
            )
            su2_sym = compute_su2_symmetry(correlation_data, num_qubits, shots)
            analysis_lines.append("")
            analysis_lines.append(r"**Symmetry Analysis (QASM)**:")
            analysis_lines.append(
                f"- Parity (Even/Odd): {parity['even']:.2f}/{parity['odd']:.2f}"
            )
            analysis_lines.append(f"- Permutation-Symmetric ZZ: {perm_sym:.2f}")
            analysis_lines.append(
                f"- SU(2) Symmetry (var): {su2_sym['su2_symmetry']:.2f}"
            )
        else:
            from . import compute_conditional_correlations, compute_su3_symmetry

            # For density matrix mode
            conditional_corrs = compute_conditional_correlations(
                density_matrix, num_qubits
            )
            avg_cc = np.mean(list(conditional_corrs.values()))
            su3_val = compute_su3_symmetry(density_matrix, num_qubits)
            analysis_lines.append("")
            analysis_lines.append(r"**Symmetry Analysis (Density)**:")
            analysis_lines.append(f"- Avg. Conditional Corr: {avg_cc:.2f}")
            analysis_lines.append(f"- SU(3) Symmetry (var): {su3_val:.2f}")

    # You could also add other CLI-based checks here, e.g. if config["plot_transitions"] is True, etc.

    # Convert lines into a single multiline string with basic styling
    # We'll do a bit of Markdown-like or ReST-like syntax
    analysis_text = "\n".join(analysis_lines)

    # Put that text in the analysis subplot (below)
    # We'll do a fancy box style with alpha=0.9
    from matplotlib.patches import FancyBboxPatch

    ax_analysis.text(
        0.01,
        0.5,
        analysis_text,
        fontsize=10,
        ha="left",
        va="center",
        # You can do more fancy styling:
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        transform=ax_analysis.transAxes,
    )

    # Hide axis lines on the analysis subplot
    ax_analysis.set_xlim(0, 1)
    ax_analysis.set_ylim(0, 1)

    # Finally save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved hypergraph to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_error_transition_graph(
    counts_list: List[Dict], time_steps: List[float], save_path: str
) -> None:
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
