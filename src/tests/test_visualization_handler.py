# src/tests/test_visualization_handler.py

import pytest
from unittest.mock import patch, Mock
from src.visualization.visualization_handler import handle_visualization
from qiskit.quantum_info import DensityMatrix
import numpy as np


def test_handle_visualization_plot_qasm_save():
    """
    Test handle_visualization with plot type, qasm mode, and save path.
    """
    result = {"counts": {"00": 512, "11": 512}}
    args = {
        "visualization_type": "plot",
        "sim_mode": "qasm",
        "min_occurrences": 0,
        "num_qubits": 2,
    }
    with patch(
        "src.visualization.visualization_handler.Visualizer.plot_histogram"
    ) as mock_plot:
        plot_closed_with_ctrl_c = handle_visualization(
            result,
            args,
            sim_mode="qasm",
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            noise_enabled=True,
            save_plot="histogram.png",
            show_plot_nonblocking=Mock(return_value=True),
        )
        assert plot_closed_with_ctrl_c is False  # No interactive plot when saving
        mock_plot.assert_called_once_with(
            result["counts"],
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            noise_enabled=True,
            save_path="histogram.png",
            min_occurrences=0,
            num_qubits=2,
        )


def test_handle_visualization_plot_qasm_interactive():
    """
    Test handle_visualization with plot type, qasm mode, and interactive display.
    """
    result = {"counts": {"00": 512, "11": 512}}
    args = {
        "visualization_type": "plot",
        "sim_mode": "qasm",
        "min_occurrences": 0,
        "num_qubits": 2,
    }
    mock_plot = Mock()

    # Define a function to simulate show_plot_nonblocking
    def mock_show_plot_nonblocking(visualizer_method, *args, **kwargs):
        visualizer_method(*args, **kwargs)  # Call the visualizer method (mock_plot)
        return True  # Simulate user pressing Enter

    with patch(
        "src.visualization.visualization_handler.Visualizer.plot_histogram", mock_plot
    ):
        plot_closed_with_ctrl_c = handle_visualization(
            result,
            args,
            sim_mode="qasm",
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            noise_enabled=True,
            save_plot=None,
            show_plot_nonblocking=mock_show_plot_nonblocking,
        )
        assert (
            plot_closed_with_ctrl_c is False
        )  # False because show_plot_nonblocking returns True
        mock_plot.assert_called_once_with(
            result["counts"],
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            noise_enabled=True,
            min_occurrences=0,
            num_qubits=2,
        )


def test_handle_visualization_hypergraph_density_save():
    """
    Test handle_visualization with hypergraph type, density mode, and save path.
    """
    result = DensityMatrix(np.array([[0.5, 0], [0, 0.5]]))
    args = {
        "visualization_type": "hypergraph",
        "sim_mode": "density",
    }
    with patch(
        "src.visualization.visualization_handler.Visualizer.plot_hypergraph"
    ) as mock_plot:
        plot_closed_with_ctrl_c = handle_visualization(
            result,
            args,
            sim_mode="density",
            state_type="GHZ",
            noise_type="DEPOLARIZING",
            noise_enabled=True,
            save_plot="hypergraph.png",
            show_plot_nonblocking=Mock(return_value=True),
        )
        assert plot_closed_with_ctrl_c is False
        mock_plot.assert_called_once()
