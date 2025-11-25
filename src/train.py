"""
Model training script with CLI interface for baseline and PyTorch models.

Usage:
    python src/train.py --data-path data/customers.parquet --model-type baseline
    python src/train.py --data-path data/customers.parquet --model-type torch
"""

import argparse
from pathlib import Path
from typing import Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, recall_score, f1_score, roc_auc_score

from config import Config
from models import RandomForestAdapter, TorchMLPAdapter


def load_features(data_path: Path) -> pd.DataFrame:
    """Load pre-computed customer features from parquet."""
    if data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")


def prepare_train_test_split(
    features_df: pd.DataFrame, config: Config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train/test with stratification to preserve class ratio.

    Returns feature matrices and labels as numpy arrays for model training.
    """
    feature_cols = [col for col in features_df.columns if col.startswith('f_')]
    X = features_df[feature_cols].values
    y = features_df['label_churn_30d'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=y,
    )

    print(f"\n[+] Train/test split:")
    print(f"    Train: {len(X_train)} samples ({y_train.mean():.1%} churn)")
    print(f"    Test:  {len(X_test)} samples ({y_test.mean():.1%} churn)")
    print(f"    Features: {X.shape[1]}")

    return X_train, X_test, y_train, y_test


def evaluate_model(
    y_true: np.ndarray, y_proba: np.ndarray, model_name: str
) -> None:
    """
    Compute and display evaluation metrics for binary classification.

    Uses recall-focused metrics aligned with churn prediction business costs.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = 0.0

    y_pred = (y_proba >= 0.5).astype(int)
    recall_at_50 = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Recall@0.5:  {recall_at_50:.4f}  (% of churners detected)")
    print(f"F1@0.5:      {f1:.4f}")
    print(f"{'='*60}\n")


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> None:
    """
    Train and evaluate Random Forest baseline.

    Uses balanced class weights to handle class imbalance.
    """
    print("\n[+] Training Random Forest baseline...")

    model = RandomForestAdapter(config)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)
    evaluate_model(y_test, y_proba, "Random Forest (Baseline)")

    print("\n[Feature Importance - Top 10]")
    feature_importance = model.get_feature_importance()
    feature_names = [
        'total_events', 'total_event_value', 'unique_event_types',
        'count_cancellation', 'count_feature_x', 'count_feature_y',
        'count_feature_z', 'count_login', 'count_page_view',
        'count_plan_downgrade', 'count_plan_upgrade', 'count_support_ticket',
        'velocity_ratio', 'login_velocity_ratio', 'days_since_last_event',
        'friction_index', 'error_rate'
    ]

    if len(feature_names) < len(feature_importance):
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]

    importance_df = pd.DataFrame({
        'feature': feature_names[:len(feature_importance)],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")


def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
) -> None:
    """
    Train and evaluate PyTorch MLP.

    Implements custom training loop with early stopping and weighted BCE loss.
    """
    print("\n[+] Training PyTorch MLP...")

    model = TorchMLPAdapter(config)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)
    evaluate_model(y_test, y_proba, "PyTorch MLP")


def main(
    data_path: Path,
    model_type: Literal['baseline', 'torch'],
) -> None:
    """
    Execute training pipeline for specified model type.

    Args:
        data_path: Path to customer features (parquet or csv)
        model_type: Either 'baseline' (Random Forest) or 'torch' (MLP)
    """
    config = Config()

    print("="*60)
    print(f"Churn Prediction Training: {model_type.upper()}")
    print("="*60)

    print(f"\n[+] Loading features from {data_path}")
    features_df = load_features(data_path)
    print(f"    Customers: {len(features_df):,}")
    print(f"    Churn rate: {features_df['label_churn_30d'].mean():.1%}")

    X_train, X_test, y_train, y_test = prepare_train_test_split(features_df, config)

    if model_type == 'baseline':
        train_baseline_model(X_train, y_train, X_test, y_test, config)
    elif model_type == 'torch':
        train_torch_model(X_train, y_train, X_test, y_test, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("\n[+] Training complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train churn prediction models on customer features'
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        required=True,
        help='Path to customer features (parquet or csv)',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['baseline', 'torch'],
        help='Model type: baseline (Random Forest) or torch (PyTorch MLP)',
    )

    args = parser.parse_args()
    main(data_path=args.data_path, model_type=args.model_type)
