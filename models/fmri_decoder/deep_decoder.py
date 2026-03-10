"""
deep_decoder.py
===============
PyTorch deep learning decoder for fMRI volumetric data.

Provides a lightweight MLP that maps a flat voxel-pattern vector to class
logits, along with a training loop suitable for offline model training before
deployment in a real-time neurofeedback pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPDecoder(nn.Module):
    """Simple MLP for fMRI pattern classification."""

    def __init__(
        self,
        n_voxels: int,
        n_classes: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = n_voxels
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepFMRIDecoder:
    """Deep learning fMRI decoder (MLP).

    Parameters
    ----------
    n_voxels : int
        Number of voxels in the input pattern (after masking).
    n_classes : int
        Number of target classes.
    hidden_dims : tuple of int
        Hidden layer sizes.
    dropout : float
        Dropout probability applied after each hidden layer.
    lr : float
        Learning rate for the Adam optimiser.
    batch_size : int
        Mini-batch size.
    epochs : int
        Number of training epochs.
    device : str or None
        Torch device (``"cuda"``, ``"cpu"``). Auto-detected when *None*.

    Examples
    --------
    >>> decoder = DeepFMRIDecoder(n_voxels=5000, n_classes=2)
    >>> decoder.fit(X_train, y_train)
    >>> probs = decoder.predict_proba(X_test)
    """

    def __init__(
        self,
        n_voxels: int,
        n_classes: int,
        hidden_dims: Tuple[int, ...] = (512, 256),
        dropout: float = 0.5,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        device: Optional[str] = None,
    ) -> None:
        self.n_voxels = n_voxels
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[_MLPDecoder] = None
        self.loss_history: list[float] = []

    def _build_model(self) -> _MLPDecoder:
        return _MLPDecoder(
            n_voxels=self.n_voxels,
            n_classes=self.n_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepFMRIDecoder":
        """Train the MLP on fMRI voxel patterns.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
        y : np.ndarray, shape (n_samples,)
            Integer class labels.

        Returns
        -------
        self
        """
        self.model = self._build_model()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        self.loss_history = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimiser.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(X_batch)

            self.loss_history.append(epoch_loss / len(dataset))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_fitted()
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        self._check_fitted()
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> "DeepFMRIDecoder":
        """Load model weights from disk."""
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
