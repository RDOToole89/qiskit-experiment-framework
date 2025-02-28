# src/visualization/hypergraph.py

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import hypernetx as hnx
from typing import Optional, Dict, List, Union, Callable
from itertools import combinations
from qiskit.quantum_info import partial_trace, Pauli, DensityMatrix
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans  # For clustering
from scipy.linalg import sqrtm  # For computing matrix square roots

logger = logging.getLogger("QuantumExperiment.Visualization")


def compute_fubini_study_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """
    Computes the Fubini-Study distance between two density matrices.

    Args:
        rho1 (np.ndarray): First density matrix.
        rho2 (np.ndarray): Second density matrix.

    Returns:
        float: Fubini-Study distance in radians.
    """
    try:
        # Compute the square root of rho1 using scipy.linalg.sqrtm
        sqrt_rho1 = sqrtm(rho1)
        # Compute inner = sqrt(rho1) @ rho2 @ sqrt(rho1)
        inner = sqrt_rho1 @ rho2 @ sqrt_rho1
        # Compute sqrt(inner) using sqrtm
        sqrt_inner = sqrtm(inner)
        fidelity = np.trace(sqrt_inner).real
        # Ensure fidelity is within [0, 1] to avoid numerical errors
        fidelity = min(max(fidelity, 0.0), 1.0)
        distance = np.arccos(fidelity)
        return distance
    except Exception as e:
        logger.error(f"Error computing Fubini-Study distance: {str(e)}")
        return 0.0  # Fallback value


def compute_pairwise_correlations(
    correlation_data: Dict, num_qubits: int, mode: str, shots: float = 1.0
) -> Dict:
    """
    Computes pairwise ZZ correlations between qubits.

    Args:
        correlation_data (Dict): Counts or density matrix data.
        num_qubits (int): Number of qubits.
        mode (str): 'qasm' or 'density'.
        shots (float): Total number of shots (for QASM mode).

    Returns:
        Dict: Pairwise correlations as a dictionary {(i,j): corr}.
    """
    correlations = {}
    if mode == "qasm":
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                zz_corr = 0.0
                for bitstring, count in correlation_data.items():
                    bit_i, bit_j = int(bitstring[i]), int(bitstring[j])
                    zz_value = (-1) ** (bit_i + bit_j)
                    zz_corr += zz_value * (count / shots)
                correlations[(i, j)] = zz_corr
    else:
        # Convert the NumPy array to a DensityMatrix object
        density_matrix = DensityMatrix(np.array(correlation_data["density"]))
        pauli_z = Pauli("Z").to_matrix()
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                # Indices to trace out: all qubits except i and j
                all_qubits = list(range(num_qubits))
                qubits_to_trace_out = [k for k in all_qubits if k not in [i, j]]
                rho_ij = partial_trace(
                    density_matrix, qargs=qubits_to_trace_out  # Trace out these qubits
                )
                # Convert rho_ij to a NumPy array for matrix multiplication
                zz_corr = np.trace(np.kron(pauli_z, pauli_z) @ rho_ij.data).real
                correlations[(i, j)] = zz_corr
    return correlations


def cluster_qubits(
    pairwise_corrs: Dict, num_qubits: int, num_clusters: int = 2
) -> List[List[int]]:
    """
    Clusters qubits based on their pairwise correlation patterns using k-means.

    Args:
        pairwise_corrs (Dict): Dictionary of pairwise correlations {(i,j): corr}.
        num_qubits (int): Number of qubits.
        num_clusters (int): Number of clusters to form (default: 2).

    Returns:
        List[List[int]]: List of clusters, where each cluster is a list of qubit indices.
    """
    # Create a feature vector for each qubit
    features = np.zeros((num_qubits, num_qubits))
    for (i, j), corr in pairwise_corrs.items():
        features[i, j] = corr
        features[j, i] = corr  # Symmetric matrix

    # Apply k-means clustering
    num_clusters = min(num_clusters, num_qubits)  # Ensure num_clusters <= num_qubits
    if num_clusters < 1:
        return [[i for i in range(num_qubits)]]  # Single cluster with all qubits
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)

    # Group qubits by cluster label
    clusters = [[] for _ in range(num_clusters)]
    for qubit_idx, label in enumerate(labels):
        clusters[label].append(qubit_idx)

    # Remove empty clusters
    clusters = [cluster for cluster in clusters if cluster]
    return clusters


def compute_su2_symmetry(counts: Dict, num_qubits: int, shots: float) -> Dict:
    """
    Computes SU(2) symmetry metrics based on ZZ correlations.

    Args:
        counts (Dict): Measurement counts.
        num_qubits (int): Number of qubits.
        shots (float): Total number of shots.

    Returns:
        Dict: SU(2) symmetry metrics.
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
    Computes Z-symmetry variance across all qubits (not true SU(3) symmetry).

    Args:
        density_matrix (np.ndarray): The density matrix of the quantum state.
        num_qubits (int): Number of qubits.

    Returns:
        float: Variance of Pauli Z expectations across qubits.
    """
    if num_qubits < 1:
        return 0.0

    # Convert to DensityMatrix object
    density_matrix = DensityMatrix(density_matrix)
    # Compute Pauli Z expectations for each qubit (up to num_qubits)
    expectations = []
    pauli_z = Pauli("Z").to_matrix()
    for qubit in range(num_qubits):  # Loop over available qubits
        # Trace out all qubits except the current qubit
        qubits_to_trace_out = [k for k in range(num_qubits) if k != qubit]
        rho_qubit = partial_trace(density_matrix, qargs=qubits_to_trace_out)
        # Compute expectation value of Pauli Z for this qubit
        expectation = np.trace(rho_qubit.data @ pauli_z).real
        expectations.append(expectation)
    # Return the variance of the Pauli Z expectations as a symmetry metric
    return np.var(expectations) if expectations else 0.0


def compute_bloch_vector(rho: Union[np.ndarray, DensityMatrix]) -> tuple:
    """
    Computes the Bloch vector for a single qubit density matrix.

    Args:
        rho (Union[np.ndarray, DensityMatrix]): Density matrix of a single qubit.

    Returns:
        tuple: (x, y, z) components of the Bloch vector.
    """
    pauli_x = Pauli("X").to_matrix()
    pauli_y = Pauli("Y").to_matrix()
    pauli_z = Pauli("Z").to_matrix()
    # If rho is a DensityMatrix, convert it to a NumPy array
    rho_array = rho.data if isinstance(rho, DensityMatrix) else rho
    x = np.trace(rho_array @ pauli_x).real
    y = np.trace(rho_array @ pauli_y).real
    z = np.trace(rho_array @ pauli_z).real
    return (x, y, z)


def plot_bloch_sphere_vectors(
    bloch_vectors: List[Dict[int, tuple]],
    time_steps: List[float],
    save_path: Optional[str],
    show_plot_nonblocking: Callable,
) -> bool:
    from mpl_toolkits.mplot3d import Axes3D

    plot_closed_with_ctrl_c = False
    num_qubits = len(
        bloch_vectors[0]
    )  # Number of qubits is the number of keys in the first timestep's dict
    for qubit in range(num_qubits):

        def plot_func():
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

        if save_path:
            plot_func()
            plt.savefig(
                f"{save_path}_bloch_qubit_{qubit}.png", bbox_inches="tight", dpi=300
            )
            logger.info(
                f"Saved Bloch sphere plot to {save_path}_bloch_qubit_{qubit}.png"
            )
            plt.close()
        else:
            print(f"Displaying Bloch sphere plot for qubit {qubit}...")
            plot_closed_with_ctrl_c |= not show_plot_nonblocking(plot_func)
    return plot_closed_with_ctrl_c


def compute_correlations(
    correlation_data: Dict,
    num_qubits: int,
    mode: str,
    config: Dict,
) -> Dict:
    """
    Computes correlations and builds hypergraph edges.

    Args:
        correlation_data (Dict): Counts or density matrix data.
        num_qubits (int): Number of qubits.
        mode (str): 'qasm' or 'density'.
        config (Dict): Visualization configuration.

    Returns:
        Dict: Hypergraph edges with weights.
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
        if "density" not in correlation_data:
            raise KeyError(
                "Expected 'density' key in correlation_data for density mode"
            )
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
    show_plot_nonblocking: Optional[Callable] = None,
) -> bool:
    """
    Plots a hypergraph of quantum state correlations with enhanced scientific visualization.

    Args:
        correlation_data: The data to plot (counts or density matrix).
        state_type: The type of quantum state.
        noise_type: The type of noise applied.
        save_path: Path to save the plot, if any.
        time_steps: Timesteps for dynamic visualization.
        config: Visualization configuration.
        show_plot_nonblocking: Function to show plots non-blockingly.

    Returns:
        bool: True if all plots were closed with Enter, False if any were closed with Ctrl+C.
    """
    config = config or {}
    config.setdefault("max_order", 2)
    config.setdefault("threshold", None)
    config.setdefault("symmetry_analysis", False)
    config.setdefault("plot_transitions", False)
    config.setdefault("plot_bloch", False)  # Ensure default is False
    config.setdefault("node_color", "blue")
    config.setdefault("edge_color", "red")

    plot_closed_with_ctrl_c = False

    # Compute Fubini-Study distances if time-stepped
    fs_distances = []
    if time_steps is not None and isinstance(correlation_data, list):
        for i in range(len(correlation_data) - 1):
            if (
                "density" in correlation_data[i]
                and "density" in correlation_data[i + 1]
            ):
                rho1 = np.array(correlation_data[i]["density"])
                rho2 = np.array(correlation_data[i + 1]["density"])
                distance = compute_fubini_study_distance(rho1, rho2)
                fs_distances.append(distance)
            else:
                fs_distances.append(0.0)  # Fallback if density data is missing

        # Plot Bloch vectors if requested and time-stepped
        if (
            config.get("plot_bloch")
            and time_steps is not None
            and isinstance(correlation_data, list)
        ):
            bloch_vectors = []
            for data in correlation_data:
                if "density" in data:
                    density_matrix = DensityMatrix(np.array(data["density"]))
                    num_qubits = int(np.log2(density_matrix.dim))
                    qubit_bloch = {}
                    for qubit in range(num_qubits):
                        qubits_to_trace_out = [
                            k for k in range(num_qubits) if k != qubit
                        ]
                        rho_qubit = partial_trace(
                            density_matrix, qargs=qubits_to_trace_out
                        )
                        bloch_vector = compute_bloch_vector(rho_qubit)
                        qubit_bloch[qubit] = bloch_vector
                    bloch_vectors.append(qubit_bloch)
            plot_closed_with_ctrl_c |= plot_bloch_sphere_vectors(
                bloch_vectors, time_steps, save_path, show_plot_nonblocking
            )

        # Plot Fubini-Study distance over time if time-stepped
        if (
            time_steps is not None
            and isinstance(correlation_data, list)
            and fs_distances
        ):

            def plot_fs_distance():
                plt.figure(figsize=(8, 6))
                plt.plot(
                    time_steps[1:],
                    fs_distances,
                    marker="o",
                    label="Fubini-Study Distance",
                    color="purple",
                )
                plt.xlabel("Time Step")
                plt.ylabel("Fubini-Study Distance (rad)")
                plt.title("Fubini-Study Distance Over Time")
                plt.legend()

            if save_path:
                plot_fs_distance()
                plt.savefig(
                    f"{save_path}_fs_distance.png", dpi=300, bbox_inches="tight"
                )
                logger.info(
                    f"Saved Fubini-Study distance plot to {save_path}_fs_distance.png"
                )
                plt.close()
            else:
                print("Displaying Fubini-Study distance plot...")
                plot_closed_with_ctrl_c |= not show_plot_nonblocking(plot_fs_distance)

        # Plot hypergraphs for each timestep
        for step, data in enumerate(correlation_data):
            fs_distance = fs_distances[step - 1] if step > 0 else None
            plot_closed_with_ctrl_c |= plot_single_hypergraph(
                data,
                state_type,
                noise_type,
                f"{save_path}_step_{step}.png" if save_path else None,
                time_steps[step],
                config,
                fs_distance=fs_distance,
                show_plot_nonblocking=show_plot_nonblocking,
            )
        if config.get("plot_transitions"):
            # Check if in QASM mode (counts_list) or density mode
            if isinstance(correlation_data, list) and all(
                isinstance(item, dict) and "density" not in item
                for item in correlation_data
            ):
                plot_closed_with_ctrl_c |= plot_error_transition_graph(
                    correlation_data, time_steps, save_path, show_plot_nonblocking
                )
            else:
                logger.warning(
                    "Skipping error transition graph in density mode as it requires QASM counts."
                )
    else:
        plot_closed_with_ctrl_c = plot_single_hypergraph(
            correlation_data,
            state_type,
            noise_type,
            save_path,
            None,
            config,
            fs_distance=None,
            show_plot_nonblocking=show_plot_nonblocking,
        )

    return plot_closed_with_ctrl_c


def plot_single_hypergraph(
    correlation_data: Dict,
    state_type: Optional[str],
    noise_type: Optional[str],
    save_path: Optional[str],
    time_step: Optional[float],
    config: Dict,
    fs_distance: Optional[float] = None,
    show_plot_nonblocking: Optional[Callable] = None,
) -> bool:
    """
    Plots a hypergraph of correlations, plus an analysis box below the plot.

    Args:
        correlation_data: The data to plot (counts or density matrix).
        state_type: The type of quantum state.
        noise_type: The type of noise applied.
        save_path: Path to save the plot, if any.
        time_step: The current timestep (if time-stepped).
        config: Visualization configuration.
        fs_distance: Fubini-Study distance for this timestep.
        show_plot_nonblocking: Function to show plots non-blockingly.

    Returns:
        bool: True if the plot was closed with Enter, False if closed with Ctrl+C.
    """
    if not correlation_data:
        logger.warning("No valid correlation data for hypergraph plotting.")
        return False

    # Distinguish QASM vs. density mode
    mode = "density" if "density" in correlation_data else "qasm"
    if mode == "density":
        if not isinstance(correlation_data, dict) or "density" not in correlation_data:
            raise KeyError(
                f"Expected a dictionary with 'density' key for density mode, got {correlation_data}"
            )
        # Convert to DensityMatrix object
        density_matrix = DensityMatrix(np.array(correlation_data["density"]))
        num_qubits = int(np.log2(density_matrix.dim))
        shots = 1.0  # Default for density mode, as shots aren't used
    else:
        if not isinstance(correlation_data, dict):
            raise TypeError(
                f"Expected a dictionary for QASM mode, got {type(correlation_data)}"
            )
        first_key = next(iter(correlation_data.keys()))
        num_qubits = len(first_key)
        shots = sum(correlation_data.values())

    # Build edges from correlation
    edges = compute_correlations(correlation_data, num_qubits, mode, config)
    if not edges:
        logger.warning("No significant correlations found for hypergraph plotting.")
        return False

    # Collect correlation values for color scaling and stats
    all_corrs = [props["weight"] for (_, props) in edges.values()]
    min_corr_val = min(all_corrs)
    max_corr_val = max(all_corrs)
    mean_corr_val = np.mean(all_corrs)
    abs_max_corr = max(abs(c) for c in all_corrs)

    # Compute pairwise correlations for clustering
    pairwise_corrs = compute_pairwise_correlations(
        correlation_data, num_qubits, mode, shots
    )

    # Cluster qubits (default to 2 clusters, adjust as needed)
    if num_qubits > 1:  # Clustering requires at least 2 qubits
        clusters = cluster_qubits(pairwise_corrs, num_qubits, num_clusters=2)
    else:
        clusters = [[0]]  # Single qubit case

    # Define the plotting function
    def plot_func():
        # Set up figure with two subplots:
        #  - top for the hypergraph
        #  - bottom for the analysis text
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
        ax_graph = fig.add_subplot(gs[0, 0])
        ax_analysis = fig.add_subplot(gs[1, 0])
        ax_analysis.set_axis_off()

        # Create a Hypernetx hypergraph
        Hedges = {
            frozenset(edge_nodes): frozenset(edge_nodes)
            for edge_key, (edge_nodes, _) in edges.items()
        }
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
        nx.draw_networkx_nodes(
            H, pos, node_color=config.get("node_color", "blue"), ax=ax_graph
        )
        nx.draw_networkx_labels(H, pos, ax=ax_graph)

        # For each hyperedge, draw a polygon
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

            # Place label at centroid
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

        # Build the analysis text
        analysis_lines = []
        analysis_lines.append(r"**Basic Correlation Stats**:")
        if mode == "qasm":
            analysis_lines.append(f"- Shots Used: {shots}")
        analysis_lines.append(f"- Min Corr: {min_corr_val:.2f}")
        analysis_lines.append(f"- Max Corr: {max_corr_val:.2f}")
        analysis_lines.append(f"- Mean Corr: {mean_corr_val:.2f}")

        # Add Fubini-Study distance if available
        if fs_distance is not None:
            analysis_lines.append("")
            analysis_lines.append(r"**Decoherence Metric**:")
            analysis_lines.append(f"- Fubini-Study Dist.: {fs_distance:.3f} rad")

        # Add clustering results
        if num_qubits > 1:
            analysis_lines.append("")
            analysis_lines.append(r"**Qubit Clustering**:")
            for idx, cluster in enumerate(clusters):
                cluster_str = ", ".join([f"q{i}" for i in cluster])
                analysis_lines.append(f"- Cluster {idx + 1}: {cluster_str}")

        # Add symmetry analysis if enabled
        if config.get("symmetry_analysis"):
            if mode == "qasm":
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
                conditional_corrs = compute_conditional_correlations(
                    density_matrix, num_qubits
                )
                avg_cc = (
                    np.mean(list(conditional_corrs.values()))
                    if conditional_corrs
                    else 0.0
                )
                su3_val = compute_su3_symmetry(density_matrix, num_qubits)
                analysis_lines.append("")
                analysis_lines.append(r"**Symmetry Analysis (Density)**:")
                analysis_lines.append(f"- Avg. Conditional Corr: {avg_cc:.2f}")
                analysis_lines.append(f"- Z-Symmetry Variance: {su3_val:.2f}")

        # Convert lines into a single multiline string
        analysis_text = "\n".join(analysis_lines)

        # Display analysis text
        ax_analysis.text(
            0.01,
            0.5,
            analysis_text,
            fontsize=10,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
            transform=ax_analysis.transAxes,
        )

        ax_analysis.set_xlim(0, 1)
        ax_analysis.set_ylim(0, 1)

    # Save or show the plot
    if save_path:
        plot_func()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved hypergraph to {save_path}")
        plt.close()
        return False  # No Ctrl+C possible when saving
    else:
        print(
            f"Displaying hypergraph for timestep {time_step if time_step is not None else 'single'}..."
        )
        return not show_plot_nonblocking(plot_func)


def plot_error_transition_graph(
    counts_list: List[Dict],
    time_steps: List[float],
    save_path: str,
    show_plot_nonblocking: Callable,
) -> bool:
    """
    Plots a transition graph showing error transitions over time (QASM mode only).

    Args:
        counts_list: List of measurement counts for each timestep.
        time_steps: List of timesteps.
        save_path: Path to save the plot, if any.
        show_plot_nonblocking: Function to show plots non-blockingly.

    Returns:
        bool: True if all plots were closed with Enter, False if any were closed with Ctrl+C.
    """
    plot_closed_with_ctrl_c = False
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

        def plot_transition():
            plt.figure(figsize=(10, 6))
            edges = [(u, v) for u, v, d in G.edges(data=True) if d["t"] == t]
            if not edges:
                plt.close()
                return
            weights = [G[u][v]["weight"] * 5 for u, v in edges]
            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
            nx.draw_networkx_labels(G, pos)
            plt.title(f"Error Transitions at t={t:.2f}")

        if save_path:
            plot_transition()
            plt.savefig(
                f"{save_path}_transition_t{t:.2f}.png", bbox_inches="tight", dpi=300
            )
            logger.info(
                f"Saved transition graph to {save_path}_transition_t{t:.2f}.png"
            )
            plt.close()
        else:
            print(f"Displaying error transition graph for t={t:.2f}...")
            plot_closed_with_ctrl_c |= not show_plot_nonblocking(plot_transition)
    return plot_closed_with_ctrl_c


def compute_parity_distribution(counts: Dict, num_qubits: int) -> Dict:
    """
    Computes the parity distribution (even/odd) of measurement outcomes.

    Args:
        counts (Dict): Measurement counts.
        num_qubits (int): Number of qubits.

    Returns:
        Dict: Parity distribution {'even': float, 'odd': float}.
    """
    parity_counts = {"even": 0, "odd": 0}
    shots = sum(counts.values())
    if shots == 0:
        return parity_counts
    for bitstring, count in counts.items():
        parity = sum(int(bit) for bit in bitstring) % 2
        parity_counts["even" if parity == 0 else "odd"] += count / shots
    return parity_counts


def compute_permutation_symmetric_correlations(
    counts: Dict, num_qubits: int, shots: float
) -> float:
    """
    Computes permutation-symmetric ZZ correlations.

    Args:
        counts (Dict): Measurement counts.
        num_qubits (int): Number of qubits.
        shots (float): Total number of shots.

    Returns:
        float: Average ZZ correlation across all pairs.
    """
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
    """
    Computes conditional ZZ correlations between pairs of qubits.

    Args:
        density_matrix (np.ndarray): Density matrix.
        num_qubits (int): Number of qubits.

    Returns:
        Dict: Conditional correlations {(i,j): corr}.
    """
    conditional_corrs = {}
    # Convert to DensityMatrix object
    density_matrix = DensityMatrix(density_matrix)
    pauli_z = Pauli("Z").to_matrix()
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Indices to trace out: all qubits except i and j
                all_qubits = list(range(num_qubits))
                qubits_to_trace_out = [k for k in all_qubits if k not in [i, j]]
                rho_ij = partial_trace(
                    density_matrix, qargs=qubits_to_trace_out  # Trace out these qubits
                )
                # Convert rho_ij to a NumPy array for matrix multiplication
                zz_corr = np.trace(np.kron(pauli_z, pauli_z) @ rho_ij.data)
                conditional_corrs[(i, j)] = zz_corr.real
    return conditional_corrs
