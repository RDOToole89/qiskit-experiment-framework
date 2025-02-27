from src.run_experiment import run_experiment
from src.visualization import plot_histogram, plot_density_matrix
import sys

num_qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 3
state_type = sys.argv[2].upper() if len(sys.argv) > 2 else "GHZ"
noise_type = sys.argv[3].upper() if len(sys.argv) > 3 else "DEPOLARIZING"
noise_enabled = sys.argv[4].strip().lower() == "true" if len(sys.argv) > 4 else True
shots = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
sim_mode = sys.argv[6].lower() if len(sys.argv) > 6 else "qasm"

result = run_experiment(num_qubits, state_type, noise_type, noise_enabled, shots, sim_mode)

if sim_mode == "density":
    plot_density_matrix(result)
else:
    plot_histogram(result, state_type, noise_type, noise_enabled)
