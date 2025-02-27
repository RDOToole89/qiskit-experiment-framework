# src/quantum_experiment/run_experiment.py

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector
from typing import Optional, Dict, Union
import logging

from src.state_preparation import prepare_state
from src.noise_models import create_noise_model

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
) -> Union[Dict, DensityMatrix]:
    qc = prepare_state(state_type, num_qubits, custom_params=custom_params, add_barrier=False)
    logger.info(f"Prepared {state_type} state with {num_qubits} qubits")

    if custom_params:
        logger.debug(f"Applied custom parameters: {custom_params}")

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
            simulate_density=(sim_mode == "density")
        )
        logger.info(
            f"Applied {noise_type} noise with params: error_rate={error_rate}, "
            f"z_prob={z_prob}, i_prob={i_prob}, t1={t1}, t2={t2}"
        )

    if sim_mode == "density":
        # TODO: Revert to method="density_matrix" once Qiskit-Aer bug with save_density_matrix is fixed
        backend = AerSimulator(method="statevector")
        qc.save_statevector()
    else:
        backend = Aer.get_backend("qasm_simulator")
        qc.measure_all()
        logger.debug("Added measurements for qasm simulation")

    circuit_compiled = transpile(qc, backend)
    print(f"Compiled circuit: {circuit_compiled}")
    print(f"Backend supports density_matrix: {backend.configuration().simulator}")
    print(f"Supported instructions: {backend.operation_names}")
    try:
        job = backend.run(
            circuit_compiled,
            shots=shots if sim_mode == "qasm" else 1,
            noise_model=noise_model,
        )
        result = job.result()
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise

    if sim_mode == "qasm":
        counts = result.get_counts()
        logger.info("Qasm simulation completed")
        return {"counts": counts, "metadata_file": "results_placeholder"}
    else:
        statevector = result.get_statevector()
        density_matrix = DensityMatrix(statevector)
        logger.info("Density simulation completed via statevector workaround")
        return density_matrix

if __name__ == "__main__":
    result = run_experiment(num_qubits=3, state_type="GHZ", sim_mode="density")
    print(result)