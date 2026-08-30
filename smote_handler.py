"""
smote_handler.py
-----------------
Module for handling class imbalance using SMOTE (Synthetic Minority
Over-sampling Technique). Applies SMOTE only on training data to
prevent data leakage.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter


class SmoteHandler:
    """
    Applies SMOTE to balance class distribution in the training set.
    
    IMPORTANT: SMOTE is applied ONLY to the training data to prevent
    data leakage into the test set.
    
    Attributes:
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize SmoteHandler.

        Args:
            random_state (int): Random seed for SMOTE. Default is 42.
        """
        self.random_state = random_state

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Apply SMOTE oversampling to the training data.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.

        Returns:
            tuple: (X_train_resampled, y_train_resampled) as DataFrames/Series.
        """
        print("\n" + "=" * 60)
        print("STEP 5: APPLYING SMOTE (ON TRAINING DATA ONLY)")
        print("=" * 60)

        # --- Display class distribution before SMOTE ---
        before_dist = Counter(y_train)
        print(f"[INFO] Class distribution BEFORE SMOTE: {dict(before_dist)}")
        for cls, count in sorted(before_dist.items()):
            print(f"       Class {cls}: {count} samples ({count / len(y_train) * 100:.1f}%)")

        try:
            # --- Apply SMOTE ---
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            # --- Convert back to DataFrame/Series for consistency ---
            X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
            y_resampled = pd.Series(y_resampled, name=y_train.name)

        except Exception as e:
            print(f"[ERROR] SMOTE failed: {e}")
            print("[INFO] Returning original training data without SMOTE.")
            return X_train, y_train

        # --- Display class distribution after SMOTE ---
        after_dist = Counter(y_resampled)
        print(f"\n[INFO] Class distribution AFTER SMOTE: {dict(after_dist)}")
        for cls, count in sorted(after_dist.items()):
            print(f"       Class {cls}: {count} samples ({count / len(y_resampled) * 100:.1f}%)")

        print(f"\n[INFO] Training set size: {len(X_train)} -> {len(X_resampled)}")
        print("-" * 60)

        return X_resampled, y_resampled
