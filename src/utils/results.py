import os
import json
import numpy as np
import logging
from qiskit.quantum_info import DensityMatrix

logger = logging.getLogger("QuantumExperiment.Utils")

def save_results(result, filename="experiment_results.json") -> None:
    """
    Saves experiment results to a JSON or NumPy file.
    """
    RESULTS_DIR = "results"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    full_filename = os.path.join(RESULTS_DIR, filename)

    if isinstance(result, dict):
        with open(full_filename, "w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved qasm results to {full_filename}")
    elif isinstance(result, DensityMatrix):
        np_filename = full_filename.rsplit(".", 1)[0] + ".npy"
        np.save(np_filename, np.array(result.data, dtype=complex))
        logger.info(f"Saved density matrix results to {np_filename}")
    else:
        raise ValueError(f"Unsupported result type: {type(result)}. Expected dict or DensityMatrix.")

    print(f"✅ Results saved to {full_filename if isinstance(result, dict) else np_filename}")

def load_results(filename="experiment_results.json"):
    """
    Loads experiment results from a JSON or NumPy file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: {filename} not found.")

    if filename.endswith(".json"):
        with open(filename, "r") as f:
            return json.load(f)
    elif filename.endswith(".npy"):
        data = np.load(filename, allow_pickle=True)
        from qiskit.quantum_info import DensityMatrix
        return DensityMatrix(data)
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use .json or .npy.")
