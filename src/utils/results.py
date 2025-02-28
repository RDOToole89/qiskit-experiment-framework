# src/utils/results.py

import os
import json
import numpy as np
import logging
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from typing import Union, Dict, Any
from datetime import datetime

logger = logging.getLogger("QuantumExperiment.Utils")


class ComplexEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle complex numbers and NumPy arrays.
    """

    def default(self, obj):
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.complex128):
            return {"real": obj.real, "imag": obj.imag}
        return json.JSONEncoder.default(self, obj)


def compute_fidelity(
    density_matrix: DensityMatrix, reference_state: Union[DensityMatrix, np.ndarray]
) -> float:
    """
    Computes the fidelity between the density matrix and a reference state.

    Args:
        density_matrix (DensityMatrix): The simulated density matrix.
        reference_state (Union[DensityMatrix, np.ndarray]): The ideal state (e.g., pure GHZ state).

    Returns:
        float: Fidelity value.
    """
    if not isinstance(reference_state, DensityMatrix):
        reference_state = DensityMatrix(reference_state)
    return state_fidelity(density_matrix, reference_state)


def get_circuit_stats(circuit: Any) -> Dict[str, int]:
    """
    Extracts basic circuit statistics.

    Args:
        circuit: QuantumCircuit object.

    Returns:
        Dict[str, int]: Statistics like depth, number of gates, etc.
    """
    return {
        "depth": circuit.depth(),
        "num_gates": sum(circuit.count_ops().values()),
        "num_qubits": circuit.num_qubits,
    }


def save_results(
    result: Union[Dict, DensityMatrix],
    experiment_params: Dict[str, Any],
    circuit: Any,
    filename: str = "experiment_results.json",
    experiment_id: str = "N/A",
) -> None:
    """
    Saves experiment results with metadata in a structured format.

    Args:
        result: Experiment result (dict for qasm, DensityMatrix for density).
        experiment_params: Parameters used in the experiment (e.g., num_qubits, noise_type).
        circuit: QuantumCircuit object used in the experiment.
        filename: Output filename.
        experiment_id: Unique identifier for the experiment.
    """
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_filename = os.path.join(RESULTS_DIR, filename)

    # Base result structure
    result_data = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": experiment_params,
        "circuit_stats": get_circuit_stats(circuit),
        "results": {},
    }

    # Handle qasm results
    if isinstance(result, dict):
        result_data["results"]["counts"] = result.get("counts", {})
        # Compute probability distribution
        total_shots = sum(result["counts"].values())
        result_data["results"]["probabilities"] = {
            state: count / total_shots for state, count in result["counts"].items()
        }
        result_data["results"]["metadata_file"] = result.get("metadata_file", "N/A")
        # Expected states for validation (e.g., GHZ should have 000 and 111)
        expected_states = (
            {"000": 0.5, "111": 0.5} if experiment_params["state_type"] == "GHZ" else {}
        )
        result_data["results"]["expected_probabilities"] = expected_states

        with open(full_filename, "w") as f:
            json.dump(result_data, f, indent=4, cls=ComplexEncoder)
        logger.info(
            f"Saved qasm results to {full_filename}",
            extra={"experiment_id": experiment_id},
        )

    # Handle density matrix results
    elif isinstance(result, DensityMatrix):
        # Save raw density matrix as NumPy array
        np_filename = full_filename.rsplit(".", 1)[0] + ".npy"
        np.save(np_filename, np.array(result.data, dtype=complex))

        # Compute fidelity against ideal state (e.g., pure GHZ state for 3 qubits)
        if experiment_params["state_type"] == "GHZ":
            ideal_state = np.zeros(2 ** experiment_params["num_qubits"], dtype=complex)
            ideal_state[0] = 1 / np.sqrt(2)  # |000⟩
            ideal_state[-1] = 1 / np.sqrt(2)  # |111⟩
            fidelity = compute_fidelity(result, ideal_state)
        else:
            fidelity = None

        # Trace out qubits to get reduced density matrices (for entanglement analysis)
        reduced_density_matrices = {}
        for qubit in range(experiment_params["num_qubits"]):
            other_qubits = list(range(experiment_params["num_qubits"]))
            other_qubits.remove(qubit)
            reduced_dm = partial_trace(result, other_qubits)
            reduced_density_matrices[f"qubit_{qubit}"] = reduced_dm.data

        result_data["results"]["density_matrix_shape"] = result.data.shape
        result_data["results"]["fidelity_to_ideal"] = fidelity
        result_data["results"]["reduced_density_matrices"] = reduced_density_matrices
        result_data["results"]["raw_data_file"] = np_filename

        with open(full_filename, "w") as f:
            json.dump(result_data, f, indent=4, cls=ComplexEncoder)
        logger.info(
            f"Saved density matrix results to {full_filename}",
            extra={"experiment_id": experiment_id},
        )

    else:
        raise ValueError(
            f"Unsupported result type: {type(result)}. Expected dict or DensityMatrix."
        )

    print(f"✅ Results saved to {full_filename}")


def load_results(filename: str) -> Dict[str, Any]:
    """
    Loads experiment results from a JSON file.

    Args:
        filename (str): Path to the results file.

    Returns:
        Dict[str, Any]: Loaded results.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found.")

    if filename.endswith(".json"):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use .json.")
