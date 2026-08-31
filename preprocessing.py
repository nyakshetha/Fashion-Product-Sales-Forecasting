"""
preprocessing.py
-----------------
Module for data preprocessing including duplicate removal,
missing value imputation, categorical standardization, and
one-hot encoding for the fitting room garment purchase dataset.
"""

import pandas as pd
import numpy as np


class Preprocessor:
    """
    Handles all preprocessing steps for the fashion purchase dataset.
    
    Steps include:
        - Removing duplicate rows
        - Imputing missing values (median for numerical, mode for categorical)
        - Normalizing numerical features (e.g., Trial_Count) to prevent dominance
        - Standardizing categorical values (lowercase, stripped whitespace)
        - One-hot encoding categorical features
    
    Attributes:
        target_column (str): Name of the target variable column.
    """

    def __init__(self, target_column: str = "Purchased"):
        """
        Initialize the Preprocessor.

        Args:
            target_column (str): Name of the target column. Default is 'Purchased'.
        """
        self.target_column = target_column

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline on the dataset.

        Args:
            data (pd.DataFrame): Raw dataset.

        Returns:
            pd.DataFrame: Cleaned and encoded dataset ready for modeling.
        """
        print("\n" + "=" * 60)
        print("STEP 2: PREPROCESSING")
        print("=" * 60)

        df = data.copy()

        # --- Step 2a: Remove duplicates ---
        df = self._remove_duplicates(df)

        # --- Step 2b: Handle missing values ---
        df = self._handle_missing_values(df)

        # --- Step 2c: Normalize numerical features to prevent dominance ---
        df = self._normalize_numerical(df)

        # --- Step 2d: Standardize categorical values ---
        df = self._standardize_categorical(df)

        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING (ONE-HOT ENCODING)")
        print("=" * 60)

        # --- Step 3: One-Hot Encoding ---
        df = self._one_hot_encode(df)

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        removed = before - after
        print(f"[INFO] Removed {removed} duplicate rows ({before} -> {after})")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values:
            - Numerical columns -> median
            - Categorical columns -> mode

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        missing_before = df.isnull().sum().sum()

        if missing_before == 0:
            print("[INFO] No missing values to impute.")
            return df

        # --- Numerical columns: fill with median ---
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"[INFO] Filled missing values in '{col}' with median: {median_val}")

        # --- Categorical columns: fill with mode ---
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"[INFO] Filled missing values in '{col}' with mode: {mode_val}")

        missing_after = df.isnull().sum().sum()
        print(f"[INFO] Missing values: {missing_before} -> {missing_after}")
        return df

    def _normalize_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features to prevent any single feature from
        dominating the model. Trial_Count is normalized to [0, 1] range.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with normalized numerical features.
        """
        # --- Normalize Trial_Count to prevent feature dominance ---
        if "Trial_Count" in df.columns:
            max_val = df["Trial_Count"].max()
            if max_val > 0:
                df["Trial_Count"] = df["Trial_Count"] / max_val
                print(f"[INFO] Normalized 'Trial_Count' (max={max_val}) to [0, 1] range")
            else:
                print("[WARNING] Trial_Count max is 0, skipping normalization.")
        else:
            print("[WARNING] 'Trial_Count' column not found, skipping normalization.")

        # --- Normalize Price to [0, 1] range to ensure fair feature contribution ---
        if "Price" in df.columns:
            max_val = df["Price"].max()
            if max_val > 0:
                df["Price"] = df["Price"] / max_val
                print(f"[INFO] Normalized 'Price' (max={max_val}) to [0, 1] range")
            else:
                print("[WARNING] Price max is 0, skipping normalization.")

        return df

    def _standardize_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical columns by converting to lowercase
        and stripping leading/trailing whitespace.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with standardized categorical values.
        """
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            df[col] = df[col].astype(str).str.lower().str.strip()
            print(f"[INFO] Standardized categorical column: '{col}'")

        print(f"[INFO] Total categorical columns standardized: {len(categorical_cols)}")
        return df

    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to all categorical features using pd.get_dummies().
        The target column is excluded from encoding.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded features.
        """
        # --- Separate target column before encoding ---
        target = df[self.target_column].copy()
        features = df.drop(columns=[self.target_column])

        # --- Identify categorical columns ---
        categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()
        print(f"[INFO] Categorical columns to encode: {categorical_cols}")

        # --- Apply one-hot encoding ---
        features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=False)

        # --- Convert boolean columns to int for compatibility ---
        bool_cols = features_encoded.select_dtypes(include=["bool"]).columns
        features_encoded[bool_cols] = features_encoded[bool_cols].astype(int)

        # --- Reattach target column ---
        df_encoded = pd.concat([features_encoded, target], axis=1)

        print(f"[INFO] Shape after one-hot encoding: {df_encoded.shape}")
        print(f"[INFO] All features are now numeric: {df_encoded.drop(columns=[self.target_column]).select_dtypes(include=[np.number]).shape[1] == df_encoded.shape[1] - 1}")
        print("-" * 60)

        return df_encoded
