"""
Model adapters for churn prediction with unified interface.

Implements both tree-based baseline (Random Forest) and deep learning
alternative (PyTorch MLP) with class-imbalance handling.
"""

from typing import Protocol, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config


class ChurnModel(Protocol):
    """
    Unified interface for heterogeneous model types.

    Enables swapping RF/MLP without changing training pipeline.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model on feature matrix and binary labels."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of churn (class 1)."""
        ...


class RandomForestAdapter:
    """
    Scikit-learn Random Forest with balanced class weights.

    Tree ensembles handle feature interactions and non-linearity naturally.
    Balanced weighting ensures minority class (churners) gets sufficient
    gradient signal despite 10:1 class imbalance.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = RandomForestClassifier(**config.rf.to_sklearn_params())

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train forest on full dataset without batching."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(churn=1) for each customer."""
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X)) if self.model.classes_[0] == 0 else np.ones(len(X))
        return proba[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """
        Gini importance for interpretability.

        Expect days_since_last_event and velocity_ratio at top.
        """
        return self.model.feature_importances_


class MLPChurnClassifier(nn.Module):
    """
    Two-hidden-layer feedforward network with BatchNorm and Dropout.

    BatchNorm stabilizes training on tabular data with varying feature scales.
    Dropout prevents overfitting despite high capacity.
    """

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout_rate: float) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (BCE loss applies sigmoid internally)."""
        return self.network(x).squeeze(-1)


class TorchMLPAdapter:
    """
    PyTorch MLP with weighted BCE loss and early stopping.

    Uses pos_weight to handle class imbalance: amplifies gradient for
    minority class (churners) by ratio of negatives to positives.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.scaler = StandardScaler()
        self.model: MLPChurnClassifier | None = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train with Adam optimizer and early stopping on validation loss.

        Splits training data 80/20 for train/val to prevent overfitting.
        """
        X_scaled = self.scaler.fit_transform(X)

        val_size = int(len(X) * 0.2)
        indices = np.random.RandomState(self.config.mlp.random_state).permutation(len(X))

        train_idx, val_idx = indices[val_size:], indices[:val_size]
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        self.model = MLPChurnClassifier(
            input_dim=X.shape[1],
            hidden_dims=self.config.mlp.hidden_dims,
            dropout_rate=self.config.mlp.dropout_rate,
        ).to(self.device)

        pos_weight = self._calculate_pos_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.mlp.learning_rate)

        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.mlp.max_epochs):
            train_loss = self._train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss = self._validate_epoch(self.model, val_loader, criterion)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= self.config.mlp.early_stopping_patience:
                break

        self.model.load_state_dict(self.best_state)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability estimates via sigmoid activation."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy()

    def _calculate_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute class imbalance ratio for loss weighting.

        Upweights churner examples to compensate for rarity.
        """
        num_neg = (y == 0).sum()
        num_pos = (y == 1).sum()
        return num_neg / (num_pos + 1e-6)

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        """Convert numpy arrays to PyTorch DataLoader for batching."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.config.mlp.batch_size,
            shuffle=shuffle,
        )

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute single training epoch with gradient updates."""
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        return total_loss / len(dataloader.dataset)

    def _validate_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Compute validation loss without gradient computation."""
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                total_loss += loss.item() * len(X_batch)

        return total_loss / len(dataloader.dataset)
