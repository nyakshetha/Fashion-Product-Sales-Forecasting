"""
evaluate.py
-----------
Module for evaluating the trained model's performance.
Includes metric computation, threshold tuning, confusion matrix,
ROC curve, Precision-Recall curve, and feature importance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import os


class Evaluator:
    """
    Evaluates model performance with comprehensive metrics and visualizations.
    
    Attributes:
        output_dir (str): Directory path to save generated plots.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the Evaluator.

        Args:
            output_dir (str): Directory to save output plots. Default is 'outputs'.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STEP 9: Threshold Tuning
    # ------------------------------------------------------------------
    def tune_threshold(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Find the optimal classification threshold by testing values from
        0.0 to 1.0 and selecting the one that maximizes the F1-score.

        Args:
            y_true (np.ndarray): True labels.
            y_proba (np.ndarray): Predicted probabilities for the positive class.

        Returns:
            float: The optimal threshold value.
        """
        print("\n" + "=" * 60)
        print("STEP 9: THRESHOLD TUNING")
        print("=" * 60)

        best_threshold = 0.5
        best_f1 = 0.0
        results = []

        # --- Test thresholds from 0.0 to 1.0 in steps of 0.01 ---
        for threshold in np.arange(0.0, 1.01, 0.01):
            y_pred_thresh = (y_proba >= threshold).astype(int)

            # Skip if all predictions are the same class (avoid undefined metrics)
            if len(np.unique(y_pred_thresh)) < 2:
                continue

            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            results.append((threshold, f1))

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # --- Default threshold comparison ---
        y_pred_default = (y_proba >= 0.5).astype(int)
        f1_default = f1_score(y_true, y_pred_default, zero_division=0)

        print(f"[INFO] Default threshold (0.50): F1-score = {f1_default:.4f}")
        print(f"[INFO] Optimal threshold ({best_threshold:.2f}): F1-score = {best_f1:.4f}")

        if best_threshold != 0.5:
            improvement = best_f1 - f1_default
            print(f"[INFO] F1-score improvement: {improvement:+.4f}")
        else:
            print("[INFO] Default threshold is already optimal.")

        print("-" * 60)
        return best_threshold

    # ------------------------------------------------------------------
    # STEP 10: Evaluation Metrics
    # ------------------------------------------------------------------
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5):
        """
        Compute and display all evaluation metrics, and generate plots.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels (using optimized threshold).
            y_proba (np.ndarray): Predicted probabilities for the positive class.
            threshold (float): The classification threshold used.
        """
        print("\n" + "=" * 60)
        print("STEP 10: MODEL EVALUATION")
        print("=" * 60)

        # --- Compute core metrics ---
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n[RESULTS] Classification Metrics (threshold = {threshold:.2f}):")
        print(f"       Accuracy  : {accuracy:.4f}")
        print(f"       Precision : {precision:.4f}")
        print(f"       Recall    : {recall:.4f}")
        print(f"       F1-Score  : {f1:.4f}")

        # --- Classification Report ---
        print(f"\n[RESULTS] Detailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Not Purchased (0)", "Purchased (1)"]))

        # --- Generate plots ---
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curve(y_true, y_proba)
        self._plot_precision_recall_curve(y_true, y_proba)

        print("-" * 60)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Generate and save a confusion matrix heatmap.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Not Purchased (0)", "Purchased (1)"],
            yticklabels=["Not Purchased (0)", "Purchased (1)"],
        )
        plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] Confusion matrix saved to: {save_path}")

    def _plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Generate and save the ROC curve plot.

        Args:
            y_true (np.ndarray): True labels.
            y_proba (np.ndarray): Predicted probabilities for the positive class.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] ROC curve saved to: {save_path}")
        print(f"[INFO] ROC AUC Score: {roc_auc:.4f}")

    def _plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Generate and save the Precision-Recall curve plot.

        Args:
            y_true (np.ndarray): True labels.
            y_proba (np.ndarray): Predicted probabilities for the positive class.
        """
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color="green", lw=2,
                 label=f"PR Curve (AP = {avg_precision:.4f})")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "precision_recall_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[INFO] Precision-Recall curve saved to: {save_path}")
        print(f"[INFO] Average Precision Score: {avg_precision:.4f}")

    # ------------------------------------------------------------------
    # STEP 11: Feature Importance
    # ------------------------------------------------------------------
    def plot_feature_importance(self, model, feature_names: list, top_n: int = 10):
        """
        Plot and save the top N most important features from the XGBoost model.
        Also runs a dominance check to verify no single feature is over-represented.

        Args:
            model: Trained XGBoost model instance.
            feature_names (list): List of feature names.
            top_n (int): Number of top features to display. Default is 10.
        """
        print("\n" + "=" * 60)
        print("STEP 11: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        # --- Get feature importances ---
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        # --- Display top features ---
        print(f"\n[RESULTS] Top {top_n} Features Influencing Purchase Prediction:")
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12} {'Percentage':<10}")
        print("-" * 65)
        total_importance = importances.sum()
        for rank, (idx, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            pct = (row['Importance'] / total_importance) * 100
            print(f"{rank:<6} {row['Feature']:<35} {row['Importance']:<12.4f} {pct:.1f}%")

        # --- Feature Dominance Check ---
        self.check_feature_dominance(importance_df, total_importance)

        # --- Plot feature importance ---
        plt.figure(figsize=(10, 7))
        top_features = importance_df.head(top_n)
        colors = ['#e74c3c' if (row['Importance'] / total_importance) > 0.4
                  else '#2ecc71' for _, row in top_features.iterrows()]
        sns.barplot(x="Importance", y="Feature", data=top_features, palette=colors)
        plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        plt.xlabel("Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n[INFO] Feature importance plot saved to: {save_path}")
        print("-" * 60)

    # ------------------------------------------------------------------
    # Feature Dominance Check
    # ------------------------------------------------------------------
    def check_feature_dominance(self, importance_df: pd.DataFrame, total_importance: float,
                                dominance_threshold: float = 0.4):
        """
        Check if any single feature dominates the model excessively.
        A feature is flagged if its importance exceeds the given threshold
        (default 40%) of total importance.

        Args:
            importance_df (pd.DataFrame): DataFrame with Feature and Importance columns.
            total_importance (float): Sum of all feature importances.
            dominance_threshold (float): Fraction above which a feature is flagged. Default 0.4.
        """
        print(f"\n[CHECK] Feature Dominance Analysis (threshold = {dominance_threshold * 100:.0f}%):")

        dominant_features = []
        for _, row in importance_df.iterrows():
            pct = row['Importance'] / total_importance
            if pct > dominance_threshold:
                dominant_features.append((row['Feature'], pct))

        if dominant_features:
            print(f"[WARNING] [!] {len(dominant_features)} feature(s) exceed {dominance_threshold * 100:.0f}% dominance:")
            for feat, pct in dominant_features:
                print(f"           - {feat}: {pct * 100:.1f}% (DOMINANT)")
            print("[WARNING] Consider additional normalization or feature engineering.")
        else:
            print("[INFO] [OK] No single feature dominates excessively.")
            print("[INFO] Feature importance is well-distributed across multiple features.")

        # --- Show distribution summary ---
        top1_pct = (importance_df.iloc[0]['Importance'] / total_importance) * 100
        top3_pct = (importance_df.head(3)['Importance'].sum() / total_importance) * 100
        top5_pct = (importance_df.head(5)['Importance'].sum() / total_importance) * 100
        print(f"\n[INFO] Importance Distribution:")
        print(f"       Top 1 feature : {top1_pct:.1f}%")
        print(f"       Top 3 features: {top3_pct:.1f}%")
        print(f"       Top 5 features: {top5_pct:.1f}%")
