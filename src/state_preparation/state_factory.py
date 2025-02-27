# src/state_preparation/state_factory.py

from qiskit import QuantumCircuit
from typing import Optional, Dict
import logging

from src.state_preparation.state_constants import STATE_CLASSES
from src.utils import logger as logger_utils

logger = logging.getLogger("QuantumExperiment.StatePreparation")

def prepare_state(
    state_type: str,
    num_qubits: int,
    custom_params: Optional[Dict] = None,
    add_barrier: bool = False,
    experiment_id: str = "N/A"
) -> QuantumCircuit:
    """
    Factory function to prepare different quantum states.
    """
    try:
        if state_type not in STATE_CLASSES:
            raise ValueError(f"Invalid state type: {state_type}. Choose from {list(STATE_CLASSES.keys())}")

        if custom_params and "custom_gates" in custom_params:
            qc = QuantumCircuit(num_qubits)
            for gate, params in custom_params["custom_gates"].items():
                qc.append(gate, params["qargs"], params.get("cargs", []))
            return qc

        state = STATE_CLASSES[state_type](num_qubits, custom_params=custom_params)
        qc = state.create(add_barrier=add_barrier, experiment_id=experiment_id)
        logger_utils.log_with_experiment_id(
            logger, "debug",
            f"Successfully created {state_type} state.",
            experiment_id,
            extra_info={"state_type": state_type, "num_qubits": num_qubits}
        )
        return qc
    except Exception as e:
        logger_utils.log_with_experiment_id(
            logger, "error",
            f"State preparation failed: {str(e)}",
            experiment_id
        )
        raise ValueError(f"State preparation failed: {str(e)}")
