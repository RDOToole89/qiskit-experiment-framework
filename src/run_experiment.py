import json
import os
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from typing import Optional, Dict, Union
from src.state_preparation import prepare_state
from src.noise_models import create_noise_model
from config.logging_config import logger
from config.experiment_config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_STATEVECTOR_DIR,
    DEFAULT_TIMESTAMP_FORMAT,
)
from config.noise_config import (
    DEFAULT_ERROR_RATE,
    DEFAULT_T1,
    DEFAULT_T2,
    DEFAULT_Z_PROB,
    DEFAULT_I_PROB,
)
from config.backend_config import get_backend


def run_experiment(
    num_qubits: int = DEFAULT_NUM_QUBITS,
    state_type: str = DEFAULT_STATE_TYPE,
    noise_type: str = DEFAULT_NOISE_TYPE,
    noise_enabled: bool = DEFAULT_NOISE_ENABLED,
    shots: int = DEFAULT_SHOTS,
    sim_mode: str = DEFAULT_SIM_MODE,
    error_rate: Optional[float] = DEFAULT_ERROR_RATE,
    z_prob: Optional[float] = DEFAULT_Z_PROB,
    i_prob: Optional[float] = DEFAULT_I_PROB,
    t1: Optional[float] = DEFAULT_T1,
    t2: Optional[float] = DEFAULT_T2,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    custom_params: Optional[Dict] = None,
) -> Union[Dict, DensityMatrix]:
    """
    Runs a quantum experiment with specified parameters, supporting extensible noise models and research analysis.
    """
    timestamp = datetime.now().strftime(DEFAULT_TIMESTAMP_FORMAT)
    statevector_filename = os.path.join(
        DEFAULT_STATEVECTOR_DIR, f"{timestamp}-{experiment_name}.npy"
    )

    # Prepare quantum circuit
    qc = prepare_state(state_type, num_qubits)
    circuit_depth = qc.depth()  # Get circuit depth for metadata
    logger.info(f"Prepared {state_type} state with {num_qubits} qubits")

    # Handle custom parameters
    if custom_params:
        logger.debug(f"Applied custom parameters: {custom_params}")

    # Apply noise if enabled
    noise_model = None
    if noise_enabled:
        noise_model = create_noise_model(
            noise_type=noise_type,
            num_qubits=num_qubits,
            error_rate=error_rate,
            z_prob=z_prob,
            i_prob=i_prob,
            t1=t1,
            t2=t2,
        )

    # Get backend
    backend = get_backend(sim_mode)

    # Compute Statevector (Before Measurement)
    statevector = Statevector.from_instruction(qc.remove_final_measurements())

    # Experiment Metadata
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "num_qubits": num_qubits,
        "state_type": state_type,
        "circuit_depth": circuit_depth,
        "sim_mode": sim_mode,
        "noise_enabled": noise_enabled,
        "noise_type": noise_type if noise_enabled else None,
        "error_rate": error_rate if noise_enabled else None,
        "t1": t1 if noise_enabled else None,
        "t2": t2 if noise_enabled else None,
    }

    # Save Statevector & Metadata
    save_statevector_to_file(statevector, statevector_filename, metadata)

    # Configure circuit for simulation mode
    if sim_mode == "qasm":
        qc.measure_all()

    # Transpile and run the circuit
    circuit_compiled = transpile(qc, backend)
    job = backend.run(
        circuit_compiled,
        shots=shots if sim_mode == "qasm" else 1,
        noise_model=noise_model,
    )
    result = job.result()

    if sim_mode == "qasm":
        counts = result.get_counts()
        logger.info(
            f"✅ Experiment '{experiment_name}' completed successfully. Results saved in {DEFAULT_STATEVECTOR_DIR}"
        )
        return {
            "counts": counts,
            "metadata_file": statevector_filename.replace(".npy", ".json"),
        }
    else:
        density_matrix = result.data(0)["density_matrix"]
        logger.info(
            f"✅ Experiment '{experiment_name}' completed successfully. Results saved in {DEFAULT_STATEVECTOR_DIR}"
        )
        return {
            "density_matrix": density_matrix,
            "statevector_file": statevector_filename,
            "metadata_file": statevector_filename.replace(".npy", ".json"),
        }


def save_statevector_to_file(
    statevector: Statevector, filename: str, metadata: dict
) -> None:
    """
    Saves the statevector to a timestamped NumPy file and stores metadata as a JSON file.

    Args:
        statevector (Statevector): The quantum statevector before measurement.
        filename (str): File path to save the statevector.
        metadata (dict): Dictionary containing experiment metadata (e.g., circuit depth, noise params).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save statevector as .npy
    np.save(filename, statevector.data)
    logger.info(f"Statevector saved to {filename}")

    # Save metadata as .json
    metadata_filename = filename.replace(".npy", ".json")
    with open(metadata_filename, "w") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    logger.info(f"Metadata saved to {metadata_filename}")
