# Quantum Experiment Simulator

An interactive, modular quantum simulation tool built with **Qiskit**. This program allows users to run quantum experiments with configurable parameters, including qubit count, quantum state type, noise models, and measurement simulations. The program also supports result visualization and logging.

---

## 🏗️ Project Structure

```
qiskit-practice/
│── quantum_experiment/       # Core quantum experiment module
│   │── __init__.py           # Module initialization
│   │── config.py             # Default configurations
│   │── noise_models.py       # Quantum noise model definitions
│   │── run_experiment.py     # Function to run quantum circuits
│   │── state_preparation.py  # State preparation (GHZ, W, G-CRY, etc.)
│   │── utils.py              # Utility functions (logging, validation, saving results)
│   │── visualization.py      # Visualization utilities (histograms, density matrix plots)
│
│── scripts/                  # Command-line scripts
│   │── run_experiment.py     # CLI script for running experiments via command-line
│
│── logs/                     # Log files directory
│
│── results/                  # Experiment results directory
│
│── main.py                   # Interactive script for running quantum experiments
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignore logs, results, cache files
```

---

## 🔧 Installation & Setup

### 1️⃣ Prerequisites

- **Python 3.8+**
- **Qiskit** (Quantum computing framework)

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

---

## 🚀 Running the Interactive Quantum Experiment

To start the interactive quantum experiment CLI, run:

```sh
python main.py
```

### 🛠️ Available Options:

When you run the script, you will be prompted to:

- Choose **qubit count**
- Select a **quantum state type** (GHZ, W, G-CRY)
- Apply **noise models** (Depolarizing, Phase Flip, etc.)
- Specify the **number of shots**
- Select **simulation mode** (QASM or Density Matrix)
- Enable **plots** (histogram or density matrix visualization)

#### Quick Start Mode (For Development)

Press `S` at the start of the program to skip input prompts and use **default settings**.

#### Rerun & Modify Settings

After an experiment:

- Press **R** to rerun the experiment with the **same parameters**.
- Press **N** to modify parameters and rerun.
- Press **Q** to **exit**.

---

## 🖥️ Running via CLI (Non-Interactive)

To run an experiment directly via CLI without interactive input, use:

```sh
python scripts/run_experiment.py --num_qubits 3 --state_type GHZ --noise_type DEPOLARIZING --shots 1024 --sim_mode qasm
```

### Available CLI Arguments:

| Argument       | Description                                      | Default        |
| -------------- | ------------------------------------------------ | -------------- |
| `--num_qubits` | Number of qubits                                 | `3`            |
| `--state_type` | Quantum state (`GHZ`, `W`, `G-CRY`)              | `GHZ`          |
| `--noise_type` | Noise model (`DEPOLARIZING`, `PHASE_FLIP`, etc.) | `DEPOLARIZING` |
| `--shots`      | Number of shots per execution                    | `1024`         |
| `--sim_mode`   | Simulation mode (`qasm` or `density`)            | `qasm`         |

---

## 📊 Results & Logs

- Experiment results are saved as JSON in `results/`
- Logs are stored in `logs/`

---

## ✨ Extending the Project

### Adding a New Quantum State

1. Open `state_preparation.py`
2. Define a new function `create_custom_state(num_qubits):`
3. Register it inside `prepare_state()`

### Adding a New Noise Model

1. Modify `noise_models.py`
2. Add a new entry inside `NOISE_FUNCTIONS`

### Modifying Visualization

1. Update `visualization.py`
2. Edit `plot_histogram()` or `plot_density_matrix()`

---

## 🎯 Future Enhancements

- Support for **hardware backend execution** (IBM Quantum)
- More **state types** and **noise models**
- **Custom quantum gates** & **circuits**

---

## 📜 License

This project is open-source and can be modified freely.

---
