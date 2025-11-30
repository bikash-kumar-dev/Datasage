#utils.py
import os
import pandas as pd
import json
from datetime import datetime
from colorama import Fore, Style

# ============================
# Safe terminal print helpers
# ============================

def info(msg):
    print(Fore.CYAN + "[INFO]" + Style.RESET_ALL, msg)

def success(msg):
    print(Fore.GREEN + "[SUCCESS]" + Style.RESET_ALL, msg)

def warn(msg):
    print(Fore.YELLOW + "[WARNING]" + Style.RESET_ALL, msg)

def error(msg):
    print(Fore.RED + "[ERROR]" + Style.RESET_ALL, msg)

# ============================
# Load dataset
# ============================

def load_csv(path):
    """Load a CSV file with safety checks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

# ============================
# Save memory/session metadata
# ============================

def save_memory(memory_dict, path="artifacts/session_memory.json"):
    """Save agent memory/session state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(memory_dict, f, indent=4)

def load_memory(path="artifacts/session_memory.json"):
    """Load previous memory if exists."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# ============================
# Timestamp helper
# ============================

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================
# Validate target column
# ============================

def validate_target(df, target):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return True
