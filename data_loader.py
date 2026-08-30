"""
data_loader.py
--------------
Module for loading and inspecting the fashion purchase dataset.
Provides the DataLoader class to read CSV files and display
basic dataset information including shape, column types, and missing values.
"""

import pandas as pd
import os


class DataLoader:
    """
    Handles loading of the fashion purchase dataset from a CSV file.
    
    Attributes:
        file_path (str): Path to the CSV file.
        data (pd.DataFrame): Loaded dataset.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with the path to the dataset.

        Args:
            file_path (str): Absolute or relative path to the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file is empty or cannot be parsed.
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)

        # --- Check if file exists ---
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"[ERROR] File not found: {self.file_path}")

        try:
            self.data = pd.read_csv(self.file_path)
            print(f"[INFO] Dataset loaded successfully from: {self.file_path}")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to read CSV file: {e}")

        # --- Validate that the dataset is not empty ---
        if self.data.empty:
            raise ValueError("[ERROR] The loaded dataset is empty.")

        # --- Display dataset information ---
        self._display_info()

        return self.data

    def _display_info(self):
        """Display basic information about the loaded dataset."""
        print(f"\n[INFO] Dataset Shape: {self.data.shape[0]} rows x {self.data.shape[1]} columns")
        print(f"\n[INFO] Column Names: {list(self.data.columns)}")

        print("\n[INFO] Data Types:")
        print(self.data.dtypes.to_string())

        print("\n[INFO] First 5 Rows:")
        print(self.data.head().to_string())

        # --- Check for missing values ---
        missing = self.data.isnull().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            print(f"\n[WARNING] Missing values detected ({total_missing} total):")
            print(missing[missing > 0].to_string())
        else:
            print("\n[INFO] No missing values found.")

        print("-" * 60)
