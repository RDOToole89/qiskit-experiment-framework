# src/config/backend_config.py

"""
📌 Backend Configuration for Qiskit Simulation
"""

from qiskit_aer import Aer, AerSimulator


def get_backend(sim_mode: str):
    """
    Returns the appropriate Qiskit backend based on the simulation mode.

    Args:
        sim_mode (str): Simulation mode ("qasm" or "density").

    Returns:
        Qiskit Backend: Selected backend for execution.
    """
    return (
        AerSimulator(method="density_matrix")
        if sim_mode == "density"
        else Aer.get_backend("qasm_simulator")
    )
