# src/run_experiment.py

"""
Run quantum experiments with specified parameters, designed for extensibility and research integration.

This module orchestrates quantum circuit preparation, noise application, simulation, and measurement,
integrating with modular components (state preparation, noise models) for a scalable "quantum experimental lab."
Supports configurable noise levels, new noise types, and logging for hypergraph correlations or fluid dynamics
analysis (e.g., structured decoherence patterns in GHZ/W states).
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector
from typing import Optional, Dict, Union, Tuple
import logging
import time
import numpy as np  # Added for trace calculation

from src.state_preparation import prepare_state
from src.noise_models import create_noise_model
from src.utils import logger as logger_utils

# Configure logger
logger = logging.getLogger("QuantumExperiment.RunExperiment")

def run_experiment(
    num_qubits: int,
    state_type: str = "GHZ",
    noise_type: str = "DEPOLARIZING",
    noise_enabled: bool = True,
    shots: int = 1024,
    sim_mode: str = "qasm",
    error_rate: Optional[float] = None,
    z_prob: Optional[float] = None,
    i_prob: Optional[float] = None,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
    custom_params: Optional[Dict] = None,
    experiment_id: str = "N/A"
) -> Tuple[QuantumCircuit, Union[Dict, DensityMatrix]]:
    """
    Runs a quantum experiment with specified parameters, supporting extensible noise models and research analysis.

    Args:
        num_qubits (int): Number of qubits in the circuit.
        state_type (str): Type of quantum state ("GHZ", "W", "CLUSTER").
        noise_type (str): Type of noise model to apply.
        noise_enabled (bool): Whether to apply noise.
        shots (int): Number of shots for qasm simulation.
        sim_mode (str): Simulation mode ("qasm" or "density").
        error_rate (float, optional): Custom error rate for noise models.
        z_prob (float, optional): Z probability for PHASE_FLIP noise.
        i_prob (float, optional): I probability for PHASE_FLIP noise.
        t1 (float, optional): T1 relaxation time for THERMAL_RELAXATION noise.
        t2 (float, optional): T2 dephasing time for THERMAL_RELAXATION noise.
        custom_params (dict, optional): Custom parameters for state preparation or noise.
        experiment_id (str): Unique identifier for this experiment run.

    Returns:
        Tuple[QuantumCircuit, Union[Dict, DensityMatrix]]: The quantum circuit and simulation result (counts for qasm, density matrix for density mode).

    Raises:
        ValueError: If invalid parameters are provided.
        Exception: If simulation fails.
    """
    # Prepare the quantum circuit
    qc = prepare_state(state_type, num_qubits, custom_params=custom_params, add_barrier=False, experiment_id=experiment_id)
    logger_utils.log_with_experiment_id(
        logger, "info",
        f"Prepared {state_type} state with {num_qubits} qubits",
        experiment_id,
        extra_info={
            "num_qubits": num_qubits,
            "state_type": state_type,
            "circuit_depth": qc.depth(),
            "num_gates": sum(qc.count_ops().values())
        }
    )

    # Apply custom parameters
    if custom_params:
        logger_utils.log_with_experiment_id(
            logger, "debug",
            f"Applied custom parameters: {custom_params}",
            experiment_id,
            extra_info={"custom_params": custom_params}
        )

    # Apply noise if enabled
    noise_model = None
    if noise_enabled:
        try:
            noise_model = create_noise_model(
                noise_type=noise_type,
                num_qubits=num_qubits,
                error_rate=error_rate,
                z_prob=z_prob,
                i_prob=i_prob,
                t1=t1,
                t2=t2,
                simulate_density=(sim_mode == "density"),
                experiment_id=experiment_id
            )
            logger_utils.log_with_experiment_id(
                logger, "info",
                (f"Applied {noise_type} noise with params: error_rate={error_rate}, "
                 f"z_prob={z_prob}, i_prob={i_prob}, t1={t1}, t2={t2}"),
                experiment_id,
                extra_info={
                    "noise_type": noise_type,
                    "error_rate": error_rate,
                    "z_prob": z_prob,
                    "i_prob": i_prob,
                    "t1": t1,
                    "t2": t2
                }
            )
        except Exception as e:
            logger_utils.log_with_experiment_id(
                logger, "error",
                f"Failed to apply noise model: {str(e)}",
                experiment_id
            )
            raise

    # Select the appropriate backend and configure the circuit
    if sim_mode == "density":
        # TODO: Revert to method="density_matrix" once Qiskit-Aer bug with save_density_matrix is fixed
        backend = AerSimulator(method="statevector")
        qc.save_statevector()
        logger_utils.log_with_experiment_id(
            logger, "debug",
            "Configured circuit for density simulation using statevector workaround",
            experiment_id
        )
    else:
        backend = Aer.get_backend("qasm_simulator")
        qc.measure_all()
        logger_utils.log_with_experiment_id(
            logger, "debug",
            "Added measurements for qasm simulation",
            experiment_id
        )

    # Transpile and run the circuit
    logger_utils.log_with_experiment_id(
        logger, "info",
        "Transpiling circuit",
        experiment_id
    )
    start_time = time.time()
    circuit_compiled = transpile(qc, backend)
    transpile_time = time.time() - start_time
    logger_utils.log_with_experiment_id(
        logger, "info",
        f"Transpilation completed in {transpile_time:.3f} seconds",
        experiment_id,
        extra_info={
            "transpile_time": transpile_time,
            "compiled_circuit_depth": circuit_compiled.depth(),
            "compiled_num_gates": sum(circuit_compiled.count_ops().values())
        }
    )
    print(f"Compiled circuit: {circuit_compiled}")
    print(f"Backend supports density_matrix: {backend.configuration().simulator}")
    print(f"Supported instructions: {backend.operation_names}")

    # Run simulation
    start_time = time.time()
    try:
        job = backend.run(
            circuit_compiled,
            shots=shots if sim_mode == "qasm" else 1,
            noise_model=noise_model,
        )
        result = job.result()
    except Exception as e:
        logger_utils.log_with_experiment_id(
            logger, "error",
            f"Simulation failed: {str(e)}",
            experiment_id
        )
        raise
    simulation_time = time.time() - start_time
    logger_utils.log_with_experiment_id(
        logger, "info",
        f"Simulation completed in {simulation_time:.3f} seconds",
        experiment_id,
        extra_info={"simulation_time": simulation_time}
    )

    # Process results
    if sim_mode == "qasm":
        counts = result.get_counts()
        total_counts = sum(counts.values())
        probabilities = {state: count / total_counts for state, count in counts.items()}
        logger_utils.log_with_experiment_id(
            logger, "info",
            "Qasm simulation completed",
            experiment_id,
            extra_info={
                "counts": counts,
                "probabilities": probabilities,
                "total_shots": total_counts
            }
        )
        result_data = {"counts": counts, "metadata_file": "results_placeholder"}
    else:
        statevector = result.get_statevector()
        density_matrix = DensityMatrix(statevector)
        logger_utils.log_with_experiment_id(
            logger, "info",
            "Density simulation completed via statevector workaround",
            experiment_id,
            extra_info={
                "density_matrix_shape": density_matrix.data.shape,
                "trace": float(np.real(np.trace(density_matrix.data)))
            }
        )
        result_data = density_matrix

    return qc, result_data


if __name__ == "__main__":
    # For testing: run an experiment in density mode
    import uuid
    experiment_id = str(uuid.uuid4())
    qc, result = run_experiment(num_qubits=3, state_type="GHZ", sim_mode="density", experiment_id=experiment_id)
    print(result)
