"""
utils.py
--------
Utility module providing helper functions for the fashion purchase
prediction pipeline, including train-test splitting and model saving.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split


class Utils:
    """
    Provides utility functions for the ML pipeline.
    """

    @staticmethod
    def split_data(
        data: pd.DataFrame,
        target_column: str = "Purchased",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split the dataset into training and test sets using stratified sampling.

        Args:
            data (pd.DataFrame): The preprocessed dataset.
            target_column (str): Name of the target column. Default is 'Purchased'.
            test_size (float): Proportion of data for testing. Default is 0.2.
            random_state (int): Random seed. Default is 42.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 60)
        print("STEP 4: TRAIN-TEST SPLIT")
        print("=" * 60)

        # --- Separate features and target ---
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # --- Stratified split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        print(f"[INFO] Total samples     : {len(data)}")
        print(f"[INFO] Training samples  : {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
        print(f"[INFO] Test samples      : {len(X_test)} ({test_size * 100:.0f}%)")
        print(f"[INFO] Stratified split  : Yes")

        # --- Show class distribution in each split ---
        train_dist = y_train.value_counts().to_dict()
        test_dist = y_test.value_counts().to_dict()
        print(f"[INFO] Training set class distribution: {train_dist}")
        print(f"[INFO] Test set class distribution    : {test_dist}")
        print("-" * 60)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_model(model, filepath: str = "outputs/purchase_model.joblib"):
        """
        Save the trained model to disk using joblib.

        Args:
            model: The trained model object to save.
            filepath (str): Path to save the model file. Default is
                            'outputs/purchase_model.joblib'.
        """
        print("\n" + "=" * 60)
        print("STEP 12: SAVING MODEL")
        print("=" * 60)

        # --- Ensure output directory exists ---
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            joblib.dump(model, filepath)
            print(f"[INFO] Trained model saved to: {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            raise

        print("-" * 60)
