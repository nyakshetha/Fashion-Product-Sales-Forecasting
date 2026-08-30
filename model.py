"""
model.py
--------
Module defining the XGBoost classification model for fashion
purchase prediction. Wraps model creation, training, and prediction.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class PurchaseModel:
    """
    XGBoost Classifier wrapper for fashion purchase prediction.
    
    Attributes:
        model (XGBClassifier): The underlying XGBoost model.
        feature_names (list): List of feature names used during training.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.08,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        """
        Initialize the PurchaseModel with XGBoost hyperparameters.

        Args:
            n_estimators (int): Number of boosting rounds. Default 200.
            max_depth (int): Maximum tree depth. Default 4.
            learning_rate (float): Step size shrinkage. Default 0.08.
            subsample (float): Fraction of samples per tree. Default 0.85.
            colsample_bytree (float): Fraction of features per tree. Default 0.85.
            reg_alpha (float): L1 regularization term. Default 1.0.
            reg_lambda (float): L2 regularization term. Default 1.0.
            random_state (int): Random seed. Default 42.
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.feature_names = None

        print("\n" + "=" * 60)
        print("STEP 6: MODEL INITIALIZATION")
        print("=" * 60)
        print(f"[INFO] XGBoost Classifier initialized with:")
        print(f"       n_estimators    = {n_estimators}")
        print(f"       max_depth       = {max_depth}")
        print(f"       learning_rate   = {learning_rate}")
        print(f"       subsample       = {subsample}")
        print(f"       colsample_bytree = {colsample_bytree}")
        print(f"       reg_alpha       = {reg_alpha}")
        print(f"       reg_lambda      = {reg_lambda}")
        print("-" * 60)

    def get_model(self) -> XGBClassifier:
        """
        Return the underlying XGBoost model instance.

        Returns:
            XGBClassifier: The XGBoost model.
        """
        return self.model

    def set_feature_names(self, feature_names: list):
        """
        Store the feature names used during training.

        Args:
            feature_names (list): List of feature column names.
        """
        self.feature_names = feature_names
        print(f"[INFO] Stored {len(feature_names)} feature names.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the given features.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels (0 or 1).
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the given features.

        Args:
            X (pd.DataFrame): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        return self.model.predict_proba(X)
