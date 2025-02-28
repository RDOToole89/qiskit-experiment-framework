# src/noise_models/phase_flip.py

from typing import Optional, List
from qiskit_aer.noise import NoiseModel, pauli_error
from .base_noise import BaseNoise

# Import the logger from base_noise
from .base_noise import logger


class PhaseFlipNoise(BaseNoise):
    """
    Phase flip noise model, applying a phase flip error to specified gates.

    Args:
        error_rate (float): The error rate for the noise model.
        num_qubits (int): Number of qubits in the circuit.
        experiment_id (str): Unique identifier for the experiment.
        z_prob (float, optional): Probability of applying a Z (phase-flip) operation.
        i_prob (float, optional): Probability of applying an I (identity) operation.
            If z_prob and i_prob are provided, they must sum to 1. If not provided,
            z_prob defaults to error_rate, and i_prob to (1 - error_rate).
    """

    def __init__(
        self,
        error_rate: float,
        num_qubits: int,
        experiment_id: str = "N/A",
        z_prob: Optional[float] = None,
        i_prob: Optional[float] = None,
    ):
        super().__init__(
            error_rate=error_rate, num_qubits=num_qubits, experiment_id=experiment_id
        )

        if z_prob is not None and i_prob is not None:
            # Validate custom probabilities if provided
            if z_prob < 0 or i_prob < 0:
                raise ValueError(
                    f"Probabilities must be non-negative: z_prob={z_prob}, i_prob={i_prob}"
                )
            total_prob = z_prob + i_prob
            if abs(total_prob - 1.0) > 1e-6:
                raise ValueError(
                    f"Probabilities must sum to 1: z_prob={z_prob}, i_prob={i_prob}, sum={total_prob}"
                )
            self.z_prob = z_prob
            self.i_prob = i_prob
        else:
            # Use error_rate as the probability of a phase flip
            self.z_prob = error_rate
            self.i_prob = 1.0 - error_rate

        # Validate that probabilities are within bounds
        if not (0 <= self.z_prob <= 1 and 0 <= self.i_prob <= 1):
            raise ValueError(
                f"Probabilities must be between 0 and 1: z_prob={self.z_prob}, i_prob={self.i_prob}"
            )

    def apply(
        self,
        noise_model: NoiseModel,
        gates: List[str],
        qubits_for_error: int = None,
        specific_qubits: Optional[List[int]] = None,
    ) -> None:
        """
        Applies phase-flip noise to the specified gates or specific qubits.

        Args:
            noise_model (NoiseModel): The noise model to add the error to.
            gates (List[str]): List of gates to apply the noise to.
            qubits_for_error (int, optional): Number of qubits the gates operate on (ignored for phase-flip noise).
            specific_qubits (List[int], optional): Specific qubits to apply noise to.
        """
        # Phase-flip noise is always 1-qubit, so apply to specific qubits if provided
        error = pauli_error([("Z", self.z_prob), ("I", self.i_prob)])
        if specific_qubits is not None:
            for qubit in specific_qubits:
                for gate in gates:
                    noise_model.add_quantum_error(error, gate, [qubit])
        else:
            # Apply to all qubits if specific_qubits is not provided
            for gate in gates:
                noise_model.add_all_qubit_quantum_error(error, gate)

        # Log the noise application using the inherited method
        self.log_noise_application(
            noise_type="PHASE_FLIP",
            gates=gates,
            extra_info={"z_prob": self.z_prob, "i_prob": self.i_prob},
        )
