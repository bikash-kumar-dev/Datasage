# agents/data_agent.py
import os
import pandas as pd
from agents.utils import load_csv

class DataAgent:
    """
    Simple DataAgent to load and provide dataset to other agents.
    """

    def __init__(self):
        self.df = None
        self.path = None

    def load(self, path: str):
        # path can be quoted or have spaces; normalize
        path = path.strip().strip('"').strip("'")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        df = load_csv(path)
        self.df = df
        self.path = path
        return df

    def get(self):
        return self.df

    def set(self, df):
        """Replace current dataset after cleaning."""
        self.df = df
        return self.df
