"""
main.py
-------
Main entry point for the Fashion Product Purchase Prediction pipeline.
Orchestrates the full ML pipeline from data loading through evaluation.

Usage:
    python main.py
"""

import os
import sys
import numpy as np
import warnings

# --- Suppress warnings for cleaner output ---
warnings.filterwarnings("ignore")

# --- Import project modules ---
from data_loader import DataLoader
from preprocessing import Preprocessor
from smote_handler import SmoteHandler
from model import PurchaseModel
from train import Trainer
from evaluate import Evaluator
from utils import Utils


def main():
    """
    Execute the complete Fashion Product Purchase Prediction pipeline.
    
    Pipeline Steps:
        1. Load dataset
        2. Preprocess data (clean, encode)
        3. Feature engineering (one-hot encoding)
        4. Train-test split (stratified 80-20)
        5. SMOTE (on training data only)
        6. Initialize XGBoost model
        7. Train model
        8. Generate predictions
        9. Threshold tuning
        10. Evaluation (metrics + plots)
        11. Feature importance analysis
        12. Save trained model
    """

    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#   FASHION PRODUCT PURCHASE PREDICTION SYSTEM           #")
    print("#" + " " * 58 + "#")
    print("#" * 60 + "\n")

    # ==========================================
    # Configuration
    # ==========================================
    DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fashion_purchase_dataset.csv")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "purchase_model.joblib")

    # Verify dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        print("[ERROR] Please place 'fashion_purchase_dataset.csv' in the project directory.")
        sys.exit(1)

    # ==========================================
    # STEP 1: Data Loading
    # ==========================================
    loader = DataLoader(file_path=DATASET_PATH)
    data = loader.load_data()

    # ==========================================
    # STEP 2 & 3: Preprocessing + Feature Engineering
    # ==========================================
    preprocessor = Preprocessor(target_column="Purchased")
    data_processed = preprocessor.preprocess(data)

    # ==========================================
    # STEP 4: Train-Test Split (Stratified 80-20)
    # ==========================================
    X_train, X_test, y_train, y_test = Utils.split_data(
        data=data_processed,
        target_column="Purchased",
        test_size=0.2,
        random_state=42,
    )

    # ==========================================
    # STEP 5: SMOTE (Training Data Only)
    # ==========================================
    smote_handler = SmoteHandler(random_state=42)
    X_train_smote, y_train_smote = smote_handler.apply_smote(X_train, y_train)

    # ==========================================
    # STEP 6: Model Initialization
    # ==========================================
    purchase_model = PurchaseModel(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
    )

    # ==========================================
    # STEP 7: Model Training
    # ==========================================
    trainer = Trainer(purchase_model=purchase_model)
    trained_model = trainer.train(X_train_smote, y_train_smote)

    # ==========================================
    # STEP 8: Generate Predictions
    # ==========================================
    y_pred_default, y_proba = trainer.get_predictions(X_test)

    # ==========================================
    # STEP 9: Threshold Tuning
    # ==========================================
    evaluator = Evaluator(output_dir=OUTPUT_DIR)
    best_threshold = evaluator.tune_threshold(y_test, y_proba)

    # --- Apply optimized threshold ---
    y_pred_optimized = (y_proba >= best_threshold).astype(int)

    # ==========================================
    # STEP 10: Evaluation
    # ==========================================
    print("\n" + "=" * 60)
    print("COMPARISON: Default vs Optimized Threshold")
    print("=" * 60)

    # --- Evaluate with default threshold (0.5) ---
    from sklearn.metrics import f1_score, accuracy_score
    acc_default = accuracy_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    print(f"[DEFAULT]   Threshold=0.50 | Accuracy={acc_default:.4f} | F1={f1_default:.4f}")

    acc_optimized = accuracy_score(y_test, y_pred_optimized)
    f1_optimized = f1_score(y_test, y_pred_optimized, zero_division=0)
    print(f"[OPTIMIZED] Threshold={best_threshold:.2f} | Accuracy={acc_optimized:.4f} | F1={f1_optimized:.4f}")
    print("-" * 60)

    # --- Full evaluation with optimized threshold ---
    evaluator.evaluate(y_test, y_pred_optimized, y_proba, threshold=best_threshold)

    # ==========================================
    # STEP 11: Feature Importance
    # ==========================================
    evaluator.plot_feature_importance(
        model=trained_model.model,
        feature_names=trained_model.feature_names,
        top_n=10,
    )

    # ==========================================
    # STEP 12: Save Model
    # ==========================================
    Utils.save_model(trained_model.model, filepath=MODEL_SAVE_PATH)

    # ==========================================
    # Final Summary
    # ==========================================
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#   PIPELINE COMPLETED SUCCESSFULLY                      #")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    print(f"\n[SUMMARY]")
    print(f"  Dataset           : {DATASET_PATH}")
    print(f"  Samples (total)   : {len(data)}")
    print(f"  Features (encoded): {X_train.shape[1]}")
    print(f"  Best Threshold    : {best_threshold:.2f}")
    print(f"  Final Accuracy    : {acc_optimized:.4f}")
    print(f"  Final F1-Score    : {f1_optimized:.4f}")
    print(f"  Model saved to    : {MODEL_SAVE_PATH}")
    print(f"  Plots saved to    : {OUTPUT_DIR}/")
    print(f"\n  Output files:")
    print(f"    - confusion_matrix.png")
    print(f"    - roc_curve.png")
    print(f"    - precision_recall_curve.png")
    print(f"    - feature_importance.png")
    print(f"    - purchase_model.joblib")
    print()


if __name__ == "__main__":
    main()
