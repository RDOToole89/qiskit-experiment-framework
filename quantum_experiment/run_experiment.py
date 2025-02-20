"""
Run quantum experiments with specified parameters, designed for extensibility and research integration.

This module orchestrates quantum circuit preparation, noise application, simulation, and measurement,
integrating with modular components (state preparation, noise models) for a scalable "quantum experimental lab."
Supports configurable noise levels, new noise types, and logging for hypergraph correlations or fluid dynamics
analysis (e.g., structured decoherence patterns in GHZ/W states).

Key features:
- Configurable noise parameters (e.g., error_rate, z_prob) for realistic or research-driven experiments.
- Scalable for multi-qubit systems (1+ qubits) and new noise types (thermal, bit flip).
- Logging of noise effects on correlations for hypergraph mapping or density matrix evolution.
- Simplified measurement logic using `measure_all()` for qasm mode.
- Extensible for custom noise, states, or backends (e.g., IBM hardware) via `custom_params`.

Example usage:
    result = run_experiment(
        num_qubits=3,
        state_type="GHZ",
        noise_type="DEPOLARIZING",
        noise_enabled=True,
        shots=1024,
        sim_mode="qasm",
        error_rate=0.1,
    )
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector
from typing import Optional, Dict, Union
import logging
from quantum_experiment.state_preparation import prepare_state
from quantum_experiment.noise_models import create_noise_model

# Configure logger for experiment-specific debugging
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
    """
    Runs a quantum experiment with specified parameters, supporting extensible noise models and research analysis.

    Prepares a quantum circuit, applies noise (if enabled), runs simulation on Qiskit backends,
    and logs effects for hypergraph or fluid dynamics insights. Supports custom parameters for
    experimental customization (e.g., hypergraph data, new backends).

    Args:
        num_qubits (int): Number of qubits in the experiment (minimum 1).
        state_type (str): Type of quantum state ("GHZ", "W", "G-CRY", default "GHZ").
        noise_type (str): Type of noise ("DEPOLARIZING", "PHASE_FLIP", etc., default "DEPOLARIZING").
        noise_enabled (bool): Whether to apply noise (default True).
        shots (int): Number of shots for qasm simulation (default 1024).
        sim_mode (str): Simulation mode ("qasm" or "density", default "qasm").
        error_rate (float, optional): Base error rate for noise models (overrides default if provided).
        z_prob (float, optional): Z probability for PHASE_FLIP noise (requires i_prob, sums to 1).
        i_prob (float, optional): I probability for PHASE_FLIP noise (requires z_prob, sums to 1).
        t1 (float, optional): T1 relaxation time for THERMAL_RELAXATION noise (seconds).
        t2 (float, optional): T2 dephasing time for THERMAL_RELAXATION noise (seconds).
        custom_params (dict, optional): Custom parameters for noise, state, or analysis customization
            (e.g., {"hypergraph_data": {"correlations": [...]}}, {"backend": "ibm"}).

    Returns:
        Union[Dict, DensityMatrix]: Simulation results (counts dict for qasm, density matrix for density mode).

    Raises:
        ValueError: If parameters are invalid or inconsistent.
    """
    # Prepare quantum circuit
    qc = prepare_state(state_type, num_qubits)
    logger.info(f"Prepared {state_type} state with {num_qubits} qubits")

    # Handle custom parameters for extensibility (e.g., hypergraph data, custom gates, new backends)
    if custom_params:
        logger.debug(f"Applied custom parameters: {custom_params}")
        # Example custom logic: modify circuit or noise based on params
        if "hypergraph_data" in custom_params:
            logger.debug(
                f"Logging correlation data for hypergraph: {custom_params['hypergraph_data']}"
            )
        if "custom_gates" in custom_params:
            for gate, params in custom_params["custom_gates"].items():
                qc.append(gate, params["qargs"], params.get("cargs", []))
        if "backend" in custom_params:
            logger.debug(f"Custom backend specified: {custom_params['backend']}")
        # Placeholder for future extensions (e.g., fluid dynamics, new noise)

    # Apply noise if enabled, with configurable parameters
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
        logger.info(
            f"Applied {noise_type} noise with params: error_rate={error_rate}, "
            f"z_prob={z_prob}, i_prob={i_prob}, t1={t1}, t2={t2}"
        )

    # Select backend based on simulation mode
    backend = (
        AerSimulator(method="density_matrix")
        if sim_mode == "density"
        else Aer.get_backend("qasm_simulator")
    )

    # Configure circuit for simulation mode
    if sim_mode == "qasm":
        # Simplified measurement for qasm mode
        qc.measure_all()
        logger.debug("Added measurements for qasm simulation")
    else:
        # Save density matrix or statevector for density/fluid analysis
        qc.save_density_matrix()
        logger.debug("Saved density matrix for density simulation")

    # Transpile and run the circuit
    circuit_compiled = transpile(qc, backend)
    job = backend.run(
        circuit_compiled,
        shots=shots if sim_mode == "qasm" else 1,
        noise_model=noise_model,
    )
    result = job.result()

    # Extract and log results, including potential correlation impacts for research
    if sim_mode == "qasm":
        counts = result.get_counts()
        logger.info(f"Qasm simulation completed: {counts}")
        # Log potential noise impact on correlations for hypergraph
        # ✅ Add hypergraph correlation logging
        hypergraph_data = {"correlations": counts}
        if noise_enabled:
            logger.debug(
                f"Noise {noise_type} may have disrupted {num_qubits}-qubit entanglement, "
                f"potentially preserving parity or symmetry patterns for hypergraph mapping."
            )
        return {"counts": counts, "hypergraph": hypergraph_data}
    else:
        density_matrix = result.data(0)["density_matrix"]
        # Optional: Save statevector for fluid dynamics (Hilbert space flow)
        statevector = Statevector.from_instruction(qc.remove_final_measurements())
        logger.info("Density simulation completed, capturing coherence loss")
        logger.debug(
            f"Density matrix evolution suggests noise as turbulence, affecting {num_qubits}-qubit flow"
        )
        return density_matrix
