# src/visualization/visualization_handler.py

from typing import Dict, Union, Optional, List, Callable
from qiskit.quantum_info import DensityMatrix
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import Visualizer
from src.visualization.hypergraph import plot_hypergraph
from src.utils import logger as logger_utils

logger = logger_utils.setup_logger(
    log_level="INFO",
    log_to_file=True,
    log_to_console=True,
    structured_log_file="logs/visualization_logs.json",
)


def handle_visualization(
    result: Union[Dict, DensityMatrix, List[Union[Dict, DensityMatrix]]],
    args: Dict,
    sim_mode: str,
    state_type: str,
    noise_type: str,
    noise_enabled: bool,
    save_plot: Optional[str],
    show_plot_nonblocking: Callable,
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
        save_plot (Optional[str]): Path to save the plot, if any.
        show_plot_nonblocking (Callable): Function to show plots non-blockingly.
        config (Optional[Dict]): Visualization configuration.
        time_steps (Optional[List[float]]): Timesteps for dynamic visualization.

    Returns:
        bool: True if the plot was closed with Enter, False if closed with Ctrl+C.
    """
    plot_closed_with_ctrl_c = False

    # For static plots (e.g., histogram, density matrix), handle differently based on whether it's stepped
    if isinstance(result, list):
        # Time-stepped simulation: process each result
        viz_results = result
        if args["visualization_type"] == "plot":
            # For density matrix plots, show each timestep
            if sim_mode == "density":
                for idx, res in enumerate(viz_results):
                    if not isinstance(res, DensityMatrix):
                        raise ValueError(
                            f"Expected a DensityMatrix object for density mode visualization, got {type(res)}"
                        )
                    args["show_real"] = args.get("show_real", False)
                    args["show_imag"] = args.get("show_imag", False)
                    timestep = time_steps[idx] if time_steps else f"step_{idx}"
                    title_suffix = (
                        f" (t={timestep:.2f})" if time_steps else f" (step {idx})"
                    )
                    logger.info(
                        f"Preparing density matrix plot for timestep {timestep}: show_real={args['show_real']}, show_imag={args['show_imag']}, save_plot={save_plot}"
                    )
                    if save_plot:
                        save_path = f"{save_plot}_timestep_{idx}.png"
                        Visualizer.plot_density_matrix(
                            res,
                            cmap="viridis",
                            show_real=args["show_real"],
                            show_imag=args["show_imag"],
                            save_path=save_path,
                            state_type=state_type,
                            noise_type=noise_type if noise_enabled else None,
                            title_suffix=title_suffix,
                        )
                        logger.info(
                            f"Saved density matrix plot to {save_path} (dimensions: {res.data.shape})"
                        )
                    else:

                        def plot_func():
                            Visualizer.plot_density_matrix(
                                res,
                                cmap="viridis",
                                show_real=args["show_real"],
                                show_imag=args["show_imag"],
                                state_type=state_type,
                                noise_type=noise_type if noise_enabled else None,
                                title_suffix=title_suffix,
                            )

                        logger.info(
                            f"Displaying density matrix plot for timestep {timestep}"
                        )
                        plot_closed_with_ctrl_c |= not show_plot_nonblocking(plot_func)
            else:  # QASM mode
                # For QASM, plot only the last result as a histogram
                single_result = viz_results[-1]
                if not isinstance(single_result, dict) or "counts" not in single_result:
                    raise ValueError(
                        "Expected a dictionary with 'counts' key for QASM mode visualization"
                    )
                if save_plot:
                    Visualizer.plot_histogram(
                        single_result["counts"],
                        state_type=state_type,
                        noise_type=noise_type if noise_enabled else None,
                        noise_enabled=noise_enabled,
                        save_path=save_path,
                        min_occurrences=args.get("min_occurrences", 0),
                        num_qubits=args["num_qubits"],
                    )
                    logger.info(f"Saved histogram plot to {save_path}")
                else:
                    logger.info("Displaying histogram plot")
                    plot_closed_with_ctrl_c = not show_plot_nonblocking(
                        Visualizer.plot_histogram,
                        single_result["counts"],
                        state_type=state_type,
                        noise_type=noise_type if noise_enabled else None,
                        noise_enabled=noise_enabled,
                        min_occurrences=args.get("min_occurrences", 0),
                        num_qubits=args["num_qubits"],
                    )
        elif args["visualization_type"] == "hypergraph":
            # Prepare correlation data for hypergraph visualization
            correlation_data = []
            for res in viz_results:
                if sim_mode == "qasm":
                    if not isinstance(res, dict) or "counts" not in res:
                        raise ValueError(
                            f"Expected a dictionary with 'counts' key for QASM mode, got {type(res)}"
                        )
                    correlation_data.append(res["counts"])
                else:  # Density mode
                    if not isinstance(res, DensityMatrix):
                        raise ValueError(
                            f"Expected a DensityMatrix object for density mode, got {type(res)}"
                        )
                    correlation_data.append({"density": res.data.tolist()})
            logger.info(
                f"Preparing hypergraph visualization with {len(correlation_data)} timesteps"
            )
            plot_closed_with_ctrl_c = not plot_hypergraph(
                correlation_data,
                state_type=state_type,
                noise_type=noise_type if noise_enabled else None,
                save_path=save_plot,
                time_steps=time_steps,
                config=config,
                show_plot_nonblocking=show_plot_nonblocking,
            )
    else:
        # Single result
        single_result = result
        if args["visualization_type"] == "plot":
            if sim_mode == "qasm":
                if not isinstance(single_result, dict) or "counts" not in single_result:
                    raise ValueError(
                        "Expected a dictionary with 'counts' key for QASM mode visualization"
                    )
                if save_plot:
                    Visualizer.plot_histogram(
                        single_result["counts"],
                        state_type=state_type,
                        noise_type=noise_type if noise_enabled else None,
                        noise_enabled=noise_enabled,
                        save_path=save_plot,
                        min_occurrences=args.get("min_occurrences", 0),
                        num_qubits=args["num_qubits"],
                    )
                    logger.info(f"Saved histogram plot to {save_plot}")
                else:
                    logger.info("Displaying histogram plot")
                    plot_closed_with_ctrl_c = not show_plot_nonblocking(
                        Visualizer.plot_histogram,
                        single_result["counts"],
                        state_type=state_type,
                        noise_type=noise_type if noise_enabled else None,
                        noise_enabled=noise_enabled,
                        min_occurrences=args.get("min_occurrences", 0),
                        num_qubits=args["num_qubits"],
                    )
            else:  # Density mode
                if not isinstance(single_result, DensityMatrix):
                    raise ValueError(
                        f"Expected a DensityMatrix object for density mode visualization, got {type(single_result)}"
                    )
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
                    logger.info(
                        f"Saved density matrix plot to {save_plot} (dimensions: {single_result.data.shape})"
                    )
                else:
                    logger.info("Displaying density matrix plot")
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
            # Single result
            if sim_mode == "qasm":
                if not isinstance(result, dict) or "counts" not in result:
                    raise ValueError(
                        f"Expected a dictionary with 'counts' key for QASM mode, got {type(result)}"
                    )
                correlation_data = result["counts"]
            else:  # Density mode
                if not isinstance(result, DensityMatrix):
                    raise ValueError(
                        f"Expected a DensityMatrix object for density mode, got {type(result)}"
                    )
                correlation_data = {"density": result.data.tolist()}

            logger.info("Preparing hypergraph visualization for single result")
            plot_closed_with_ctrl_c = not plot_hypergraph(
                correlation_data,
                state_type=state_type,
                noise_type=noise_type if noise_enabled else None,
                save_path=save_plot,
                time_steps=time_steps,
                config=config,
                show_plot_nonblocking=show_plot_nonblocking,
            )

    return plot_closed_with_ctrl_c
