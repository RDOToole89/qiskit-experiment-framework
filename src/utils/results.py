# src/quantum_experiment/utils/results.py

import os
import json
import numpy as np
from qiskit.quantum_info import DensityMatrix
from src.config.paths_config import RESULTS_DIR


def save_results(result, filename="experiment_results.json"):
    """Saves experiment results in JSON or NumPy format."""
    full_filename = os.path.join(RESULTS_DIR, filename)

    if isinstance(result, dict):  # QASM counts
        with open(full_filename, "w") as f:
            json.dump(result, f, indent=4)
    elif isinstance(result, DensityMatrix):  # Density matrix
        np_filename = full_filename.rsplit(".", 1)[0] + ".npy"
        np.save(np_filename, np.array(result.data, dtype=complex))
    else:
        raise ValueError("Unsupported result type: Expected Dict or DensityMatrix.")

    print(f"✅ Results saved to {full_filename}")


def load_results(filename="experiment_results.json"):
    """Loads experiment results from JSON or NumPy file."""
    full_filename = os.path.join(RESULTS_DIR, filename)

    if filename.endswith(".json"):
        with open(full_filename, "r") as f:
            return json.load(f)
    elif filename.endswith(".npy"):
        return DensityMatrix(np.load(full_filename, allow_pickle=True))
    else:
        raise ValueError("Unsupported file format: Use .json or .npy")
