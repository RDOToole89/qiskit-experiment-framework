# src/config/defaults.py

import configparser
import os

# Ensure config.ini exists
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

# Load configuration
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

# Defaults
DEFAULT_NUM_QUBITS = int(config["DEFAULT"]["NUM_QUBITS"])
DEFAULT_STATE_TYPE = config["DEFAULT"]["STATE_TYPE"]
DEFAULT_NOISE_TYPE = config["DEFAULT"]["NOISE_TYPE"]
DEFAULT_NOISE_ENABLED = config["DEFAULT"].getboolean("NOISE_ENABLED")
DEFAULT_SHOTS = int(config["DEFAULT"]["SHOTS"])
DEFAULT_SIM_MODE = config["DEFAULT"]["SIM_MODE"]
DEFAULT_ERROR_RATE = float(config["DEFAULT"]["ERROR_RATE"])
