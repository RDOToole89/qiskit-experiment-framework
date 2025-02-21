# src/config/paths.py

""" File paths configuration for experiment logs and results """

import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Directories
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATEVECTOR_DIR = os.path.join(RESULTS_DIR, "statevector")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")  # ✅ Add this line

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATEVECTOR_DIR, exist_ok=True)
