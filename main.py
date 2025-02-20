# src/main.py

"""
Interactive script to run quantum experiments.
"""

import matplotlib.pyplot as plt
from datetime import datetime
from quantum_experiment.run_experiment import run_experiment
from quantum_experiment.utils import setup_logger, validate_inputs, save_results
from quantum_experiment.visualization import plot_histogram, plot_density_matrix
from quantum_experiment.config import (
    DEFAULT_NUM_QUBITS,
    DEFAULT_STATE_TYPE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_ENABLED,
    DEFAULT_SHOTS,
    DEFAULT_SIM_MODE,
)

# Configure logger
logger = setup_logger()


def interactive_experiment():
    """Runs the quantum experiment interactively with rerun and skip options."""

    while True:
        print("\n🚀 Welcome to the Quantum Experiment Interactive Runner!")
        print("🔹 Choose an option:")
        print("🔄 Press 'S' to **skip** and use default settings")
        print("🆕 Press 'N' to **enter parameters manually**")
        print("❌ Press 'Q' to **quit**")

        user_choice = input("➡️ Your choice: ").strip().upper()

        if user_choice == "S":
            print("\n⚡ Skipping input. Running with **default configuration**...\n")
            num_qubits = DEFAULT_NUM_QUBITS
            state_type = DEFAULT_STATE_TYPE
            noise_type = DEFAULT_NOISE_TYPE
            noise_enabled = DEFAULT_NOISE_ENABLED
            shots = DEFAULT_SHOTS
            sim_mode = DEFAULT_SIM_MODE
            show_plot = True  # Default plot enabled
        elif user_choice == "N":
            print("\n🔹 Choose your experiment parameters below:\n")
            num_qubits = int(
                input(f"Enter number of qubits [{DEFAULT_NUM_QUBITS}]: ")
                or DEFAULT_NUM_QUBITS
            )
            state_type = (
                input(f"Enter state type (GHZ/W/G-CRY) [{DEFAULT_STATE_TYPE}]: ")
                or DEFAULT_STATE_TYPE
            )
            noise_type = (
                input(
                    f"Enter noise type (DEPOLARIZING/PHASE_FLIP/AMPLITUDE_DAMPING/PHASE_DAMPING) [{DEFAULT_NOISE_TYPE}]: "
                )
                or DEFAULT_NOISE_TYPE
            )
            noise_enabled = (
                input(f"Enable noise? (true/false) [{DEFAULT_NOISE_ENABLED}]: ")
                .strip()
                .lower()
                == "true"
            )
            shots = int(
                input(f"Enter number of shots [{DEFAULT_SHOTS}]: ") or DEFAULT_SHOTS
            )
            sim_mode = (
                input(f"Enter simulation mode (qasm/density) [{DEFAULT_SIM_MODE}]: ")
                or DEFAULT_SIM_MODE
            )
            show_plot = (
                input(f"Show plot? (true/false) [True]: ").strip().lower() == "true"
            )
        elif user_choice == "Q":
            print("\n👋 Exiting Quantum Experiment Runner. Goodbye!")
            return
        else:
            print("⚠️ Invalid choice! Please enter S, N, or Q.")
            continue

        while True:
            logger.info("Validating inputs...")
            validate_inputs(
                num_qubits,
                state_type,
                noise_type,
                sim_mode,
                valid_states=["GHZ", "W", "G-CRY"],
                valid_noises=[
                    "DEPOLARIZING",
                    "PHASE_FLIP",
                    "AMPLITUDE_DAMPING",
                    "PHASE_DAMPING",
                ],
                valid_modes=["qasm", "density"],
            )

            logger.info(
                f"Starting experiment with {num_qubits} qubits, {state_type} state, "
                f"{'with' if noise_enabled else 'without'} {noise_type} noise."
            )

            result = run_experiment(
                num_qubits, state_type, noise_type, noise_enabled, shots, sim_mode
            )

            # Save results
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            result_filename = f"results/experiment_results_{num_qubits}q_{state_type}_{noise_type}_{sim_mode}_{timestamp}.json"
            save_results(result.to_dict(), result_filename)

            logger.info(
                f"✅ Experiment completed! Results saved in `{result_filename}`\n"
            )

            print(
                f"\n✅ Experiment completed successfully!\n📁 Results saved in `{result_filename}`"
            )

            # **Clear old plots before rerunning**
            plt.close("all")

            # **Show plot if requested**
            if show_plot and sim_mode == "qasm":
                plot_histogram(
                    result.get_counts(), state_type, noise_type, noise_enabled
                )
            elif show_plot and sim_mode == "density":
                plot_density_matrix(result.data(0)["density_matrix"])

            # **Ask for next action**
            print("\n🔄 Press 'R' to rerun with **same parameters**")
            print("🔄 Press 'N' to rerun with **new parameters**")
            print("❌ Press 'Q' to **quit**")

            user_choice = input("➡️ Your choice: ").strip().upper()

            if user_choice == "R":
                print("\n🔁 Rerunning experiment with **same parameters**...\n")
                continue  # Rerun with same parameters
            elif user_choice == "N":
                print("\n🆕 Rerunning with **new parameters**...\n")
                break  # Restart parameter input
            elif user_choice == "Q":
                print("\n👋 Exiting Quantum Experiment Runner. Goodbye!")
                return
            else:
                print("⚠️ Invalid choice! Please enter R, N, or Q.")


if __name__ == "__main__":
    interactive_experiment()
