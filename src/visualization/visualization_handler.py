# src/visualization/visualization_handler.py

from typing import Dict, Union, Optional, List
from qiskit.quantum_info import DensityMatrix
import numpy as np
from src.visualization import Visualizer
from src.visualization.hypergraph import plot_hypergraph


def handle_visualization(
    result: Union[Dict, DensityMatrix, List[Union[Dict, DensityMatrix]]],
    args: Dict,
    sim_mode: str,
    state_type: str,
    noise_type: str,
    noise_enabled: bool,
    save_plot: str,
    show_plot_nonblocking: callable,
    config: Optional[Dict] = None,
    time_steps: Optional[List[float]] = None,
) -> bool:
    """
    Handles visualization based on the specified visualization type.

    Args:
        result (Union[Dict, DensityMatrix, List[Union[Dict, DensityMatrix]]]):
            The simulation result or a list of simulation results.
        args (Dict): Experiment parameters.
        sim_mode (str): Simulation mode ('qasm' or 'density').
        state_type (str): The type of quantum state.
        noise_type (str): The type of noise applied.
        noise_enabled (bool): Whether noise is enabled.
        save_plot (str): Path to save the plot, if any.
        show_plot_nonblocking (callable): Function to show plots non-blockingly.
        config (Dict, optional): Visualization configuration.
        time_steps (List[float], optional): Timesteps for dynamic visualization.

    Returns:
        bool: True if the plot was closed with Enter, False if closed with Ctrl+C.
    """
    plot_closed_with_ctrl_c = False

    # If result is a list, select the last result for static plots.
    if isinstance(result, list):
        single_result = result[-1]
    else:
        single_result = result

    if args["visualization_type"] == "plot":
        if sim_mode == "qasm":
            if save_plot:
                Visualizer.plot_histogram(
                    single_result["counts"],
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
                    single_result["counts"],
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                    noise_enabled=noise_enabled,
                    min_occurrences=args["min_occurrences"],
                    num_qubits=args["num_qubits"],
                )
        else:
            # For density mode, set display preferences.
            args["show_real"] = args.get("show_real", False)
            args["show_imag"] = args.get("show_imag", False)
            if save_plot:
                Visualizer.plot_density_matrix(
                    single_result,
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
                    single_result,
                    cmap="viridis",
                    show_real=args["show_real"],
                    show_imag=args["show_imag"],
                    state_type=state_type,
                    noise_type=noise_type if noise_enabled else None,
                )
    elif args["visualization_type"] == "hypergraph":
        # For hypergraph visualization, if in qasm mode use counts,
        # otherwise, use density matrix data or hypergraph correlations if available.
        if sim_mode == "qasm":
            correlation_data = single_result["counts"]
        else:
            if isinstance(single_result, DensityMatrix):
                correlation_data = {"density": np.abs(single_result.data).tolist()}
            elif isinstance(single_result, dict):
                correlation_data = single_result.get("hypergraph", {}).get(
                    "correlations", {}
                )
            else:
                correlation_data = {}
        plot_hypergraph(
            correlation_data,
            state_type=state_type,
            noise_type=noise_type if noise_enabled else None,
            save_path=save_plot,
            time_steps=time_steps,
            config=config,
        )
        # For hypergraph visualization, we assume non-blocking behavior.
    return plot_closed_with_ctrl_c
