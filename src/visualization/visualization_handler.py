# src/visualization/visualization_handler.py

from typing import Dict, Union
from qiskit.quantum_info import DensityMatrix
import numpy as np
from src.visualization import Visualizer


def handle_visualization(
    result: Union[Dict, DensityMatrix],
    args: Dict,
    sim_mode: str,
    state_type: str,
    noise_type: str,
    noise_enabled: bool,
    save_plot: str,
    show_plot_nonblocking: callable,
) -> bool:
    """
    Handles visualization based on the specified visualization type.

    Args:
        result (Union[Dict, DensityMatrix]): The simulation result.
        args (Dict): Experiment parameters.
        sim_mode (str): Simulation mode ('qasm' or 'density').
        state_type (str): The type of quantum state.
        noise_type (str): The type of noise applied.
        noise_enabled (bool): Whether noise is enabled.
        save_plot (str): Path to save the plot, if any.
        show_plot_nonblocking (callable): Function to show plots non-blockingly.

    Returns:
        bool: True if the plot was closed with Enter, False if closed with Ctrl+C.
    """
    plot_closed_with_ctrl_c = False
    if args["visualization_type"] == "plot":
        if sim_mode == "qasm":
            if save_plot:
                Visualizer.plot_histogram(
                    result["counts"],
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                    noise_enabled=noise_enabled,
                    save_path=save_plot,
                    min_occurrences=args["min_occurrences"],
                    num_qubits=args["num_qubits"],
                )
            else:
                plot_closed_with_ctrl_c = not show_plot_nonblocking(
                    Visualizer.plot_histogram,
                    result["counts"],
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                    noise_enabled=noise_enabled,
                    min_occurrences=args["min_occurrences"],
                    num_qubits=args["num_qubits"],
                )
        else:
            args["show_real"] = args.get("show_real", False)
            args["show_imag"] = args.get("show_imag", False)
            if save_plot:
                Visualizer.plot_density_matrix(
                    result,
                    cmap="viridis",
                    show_real=args["show_real"],
                    show_imag=args["show_imag"],
                    save_path=save_plot,
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                )
            else:
                plot_closed_with_ctrl_c = not show_plot_nonblocking(
                    Visualizer.plot_density_matrix,
                    result,
                    cmap="viridis",
                    show_real=args["show_real"],
                    show_imag=args["show_imag"],
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                )
    elif args["visualization_type"] == "hypergraph":
        correlation_data = (
            result["counts"]
            if sim_mode == "qasm"
            else (
                {"density": np.abs(result.data).tolist()}
                if isinstance(result, DensityMatrix)
                else (
                    result.get("hypergraph", {}).get("correlations", {})
                    if isinstance(result, dict)
                    else {}
                )
            )
        )
        if save_plot:
            Visualizer.plot_hypergraph(
                correlation_data,
                state_type=state_type,
                noise_type=noise_type if noise_enabled else None,
                save_path=save_plot,
            )
        else:
            plot_closed_with_ctrl_c = not show_plot_nonblocking(
                Visualizer.plot_hypergraph,
                correlation_data,
                state_type=state_type,
                noise_type=noise_type if noise_enabled else None,
            )

    return plot_closed_with_ctrl_c
