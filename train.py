"""
train.py
--------
Module for training the XGBoost model on the preprocessed and
SMOTE-balanced training data.
"""

import pandas as pd
import numpy as np
from model import PurchaseModel


class Trainer:
    """
    Handles the training process for the purchase prediction model.
    
    Attributes:
        purchase_model (PurchaseModel): The model wrapper instance.
    """

    def __init__(self, purchase_model: PurchaseModel):
        """
        Initialize the Trainer with a PurchaseModel instance.

        Args:
            purchase_model (PurchaseModel): The initialized model wrapper.
        """
        self.purchase_model = purchase_model

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the XGBoost model on the provided training data.

        Args:
            X_train (pd.DataFrame): Training feature matrix.
            y_train (pd.Series): Training target labels.

        Returns:
            PurchaseModel: The trained model wrapper.
        """
        print("\n" + "=" * 60)
        print("STEP 7: MODEL TRAINING")
        print("=" * 60)

        # --- Store feature names for later use ---
        self.purchase_model.set_feature_names(list(X_train.columns))

        print(f"[INFO] Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

        try:
            # --- Fit the model ---
            self.purchase_model.model.fit(X_train, y_train)
            print("[INFO] Model training completed successfully.")
        except Exception as e:
            print(f"[ERROR] Model training failed: {e}")
            raise

        print("-" * 60)
        return self.purchase_model

    def get_predictions(self, X_test: pd.DataFrame):
        """
        Generate predictions and probability scores on the test set.

        Args:
            X_test (pd.DataFrame): Test feature matrix.

        Returns:
            tuple: (y_pred, y_proba) where y_pred are class labels
                   and y_proba are probability scores for the positive class.
        """
        print("\n" + "=" * 60)
        print("STEP 8: GENERATING PREDICTIONS")
        print("=" * 60)

        # --- Predict labels ---
        y_pred = self.purchase_model.predict(X_test)
        print(f"[INFO] Predicted labels generated for {len(y_pred)} samples.")

        # --- Predict probabilities ---
        y_proba_all = self.purchase_model.predict_proba(X_test)
        y_proba = y_proba_all[:, 1]  # Probability of positive class (Purchased = 1)
        print(f"[INFO] Probability scores generated.")
        print(f"       Mean probability: {y_proba.mean():.4f}")
        print(f"       Min probability:  {y_proba.min():.4f}")
        print(f"       Max probability:  {y_proba.max():.4f}")

        print("-" * 60)
        return y_pred, y_proba
