"""
cross_modal_model.py
====================
Cross-modal neural representation mapping between fMRI and EEG/fNIRS.

Implements three complementary approaches:

1. **CCAMapper** – Canonical Correlation Analysis for linear alignment of
   fMRI and EEG/fNIRS feature spaces.

2. **DeepCrossModalMapper** – A pair of PyTorch encoder networks trained with
   a contrastive (NT-Xent) loss to learn shared latent representations.

3. **CrossModalMapper** – Unified interface exposing both methods.

Research context
----------------
The central research question (RQ1) asks whether neural representations decoded
from fMRI can be predicted from EEG/fNIRS. These models provide the computational
bridge required to investigate that question.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Canonical Correlation Analysis mapper
# ---------------------------------------------------------------------------

class CCAMapper:
    """Linear cross-modal mapping via Canonical Correlation Analysis.

    Finds a shared low-dimensional subspace that maximally correlates
    the fMRI representation with the EEG/fNIRS feature vector.

    Parameters
    ----------
    n_components : int
        Number of canonical variates to retain.
    scale : bool
        Whether to z-score both modalities before fitting.

    Examples
    --------
    >>> mapper = CCAMapper(n_components=10)
    >>> mapper.fit(fmri_features, eeg_features)
    >>> fmri_shared, eeg_shared = mapper.transform(fmri_test, eeg_test)
    """

    def __init__(self, n_components: int = 10, scale: bool = True) -> None:
        self.n_components = n_components
        self.scale = scale

        self._cca: Optional[CCA] = None
        self._scaler_fmri: Optional[StandardScaler] = None
        self._scaler_eeg: Optional[StandardScaler] = None

    def fit(self, X_fmri: np.ndarray, X_eeg: np.ndarray) -> "CCAMapper":
        """Fit CCA on paired fMRI and EEG/fNIRS features.

        Parameters
        ----------
        X_fmri : np.ndarray, shape (n_samples, n_fmri_features)
        X_eeg  : np.ndarray, shape (n_samples, n_eeg_features)

        Returns
        -------
        self
        """
        if self.scale:
            self._scaler_fmri = StandardScaler()
            self._scaler_eeg = StandardScaler()
            X_fmri = self._scaler_fmri.fit_transform(X_fmri)
            X_eeg = self._scaler_eeg.fit_transform(X_eeg)

        self._cca = CCA(n_components=self.n_components)
        self._cca.fit(X_fmri, X_eeg)
        return self

    def transform(
        self, X_fmri: np.ndarray, X_eeg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project both modalities into the shared canonical space.

        Parameters
        ----------
        X_fmri : np.ndarray, shape (n_samples, n_fmri_features)
        X_eeg  : np.ndarray, shape (n_samples, n_eeg_features)

        Returns
        -------
        (fmri_canonical, eeg_canonical) : Tuple of np.ndarray,
            each shape (n_samples, n_components)
        """
        self._check_fitted()
        if self.scale:
            X_fmri = self._scaler_fmri.transform(X_fmri)
            X_eeg = self._scaler_eeg.transform(X_eeg)
        return self._cca.transform(X_fmri, X_eeg)

    def predict_eeg_from_fmri(self, X_fmri: np.ndarray) -> np.ndarray:
        """Predict EEG/fNIRS canonical variates from fMRI features.

        Parameters
        ----------
        X_fmri : np.ndarray, shape (n_samples, n_fmri_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
        """
        self._check_fitted()
        if self.scale:
            X_fmri = self._scaler_fmri.transform(X_fmri)
        return X_fmri @ self._cca.x_rotations_

    def _check_fitted(self) -> None:
        if self._cca is None:
            raise RuntimeError("CCAMapper has not been fitted. Call fit() first.")


# ---------------------------------------------------------------------------
# Deep cross-modal encoder (contrastive learning)
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """Small MLP encoder projecting modality features into a shared embedding."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class DeepCrossModalMapper:
    """Deep cross-modal mapper trained with NT-Xent contrastive loss.

    Trains a pair of encoders (one per modality) to map fMRI and EEG/fNIRS
    features into a shared normalised embedding space where paired samples
    are attracted and unpaired samples are repelled.

    Parameters
    ----------
    n_fmri_features : int
        Dimensionality of the fMRI feature vector.
    n_eeg_features : int
        Dimensionality of the EEG/fNIRS feature vector.
    embed_dim : int
        Dimensionality of the shared embedding space.
    hidden_dim : int
        Hidden layer size in each encoder.
    temperature : float
        NT-Xent temperature parameter.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    epochs : int
        Number of training epochs.
    device : str or None
        Torch device.

    Examples
    --------
    >>> mapper = DeepCrossModalMapper(n_fmri_features=100, n_eeg_features=64)
    >>> mapper.fit(fmri_features, eeg_features)
    >>> embeddings = mapper.encode_fmri(fmri_test)
    """

    def __init__(
        self,
        n_fmri_features: int,
        n_eeg_features: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        temperature: float = 0.07,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        device: Optional[str] = None,
    ) -> None:
        self.n_fmri_features = n_fmri_features
        self.n_eeg_features = n_eeg_features
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.fmri_encoder: Optional[_Encoder] = None
        self.eeg_encoder: Optional[_Encoder] = None
        self.loss_history: list[float] = []

    def _nt_xent_loss(
        self, z_fmri: torch.Tensor, z_eeg: torch.Tensor
    ) -> torch.Tensor:
        """NT-Xent contrastive loss for a batch of paired embeddings."""
        n = z_fmri.shape[0]
        z = torch.cat([z_fmri, z_eeg], dim=0)  # (2n, embed_dim)
        sim = torch.mm(z, z.T) / self.temperature  # (2n, 2n)

        # Mask out self-similarity
        mask = torch.eye(2 * n, dtype=torch.bool, device=sim.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+n) and (i+n, i)
        labels = torch.cat(
            [torch.arange(n, 2 * n), torch.arange(0, n)]
        ).to(sim.device)
        return F.cross_entropy(sim, labels)

    def fit(self, X_fmri: np.ndarray, X_eeg: np.ndarray) -> "DeepCrossModalMapper":
        """Train both encoders on paired fMRI and EEG/fNIRS features.

        Parameters
        ----------
        X_fmri : np.ndarray, shape (n_samples, n_fmri_features)
        X_eeg  : np.ndarray, shape (n_samples, n_eeg_features)

        Returns
        -------
        self
        """
        self.fmri_encoder = _Encoder(self.n_fmri_features, self.hidden_dim, self.embed_dim).to(
            self.device
        )
        self.eeg_encoder = _Encoder(self.n_eeg_features, self.hidden_dim, self.embed_dim).to(
            self.device
        )

        optimiser = torch.optim.Adam(
            list(self.fmri_encoder.parameters()) + list(self.eeg_encoder.parameters()),
            lr=self.lr,
        )

        fmri_t = torch.tensor(X_fmri, dtype=torch.float32)
        eeg_t = torch.tensor(X_eeg, dtype=torch.float32)
        dataset = TensorDataset(fmri_t, eeg_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.fmri_encoder.train()
        self.eeg_encoder.train()
        self.loss_history = []

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for fmri_batch, eeg_batch in loader:
                fmri_batch = fmri_batch.to(self.device)
                eeg_batch = eeg_batch.to(self.device)

                z_fmri = self.fmri_encoder(fmri_batch)
                z_eeg = self.eeg_encoder(eeg_batch)

                loss = self._nt_xent_loss(z_fmri, z_eeg)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss / max(len(loader), 1))

        return self

    def encode_fmri(self, X_fmri: np.ndarray) -> np.ndarray:
        """Encode fMRI features into the shared embedding space.

        Parameters
        ----------
        X_fmri : np.ndarray, shape (n_samples, n_fmri_features)

        Returns
        -------
        np.ndarray, shape (n_samples, embed_dim)
        """
        self._check_fitted()
        self.fmri_encoder.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_fmri, dtype=torch.float32).to(self.device)
            return self.fmri_encoder(x_t).cpu().numpy()

    def encode_eeg(self, X_eeg: np.ndarray) -> np.ndarray:
        """Encode EEG/fNIRS features into the shared embedding space.

        Parameters
        ----------
        X_eeg : np.ndarray, shape (n_samples, n_eeg_features)

        Returns
        -------
        np.ndarray, shape (n_samples, embed_dim)
        """
        self._check_fitted()
        self.eeg_encoder.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_eeg, dtype=torch.float32).to(self.device)
            return self.eeg_encoder(x_t).cpu().numpy()

    def _check_fitted(self) -> None:
        if self.fmri_encoder is None or self.eeg_encoder is None:
            raise RuntimeError(
                "DeepCrossModalMapper has not been fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

class CrossModalMapper:
    """Unified cross-modal mapping interface.

    Wraps either CCAMapper or DeepCrossModalMapper with a consistent API.

    Parameters
    ----------
    method : str
        ``"cca"`` (default) or ``"deep"``.
    **kwargs
        Forwarded to the underlying mapper.

    Examples
    --------
    >>> mapper = CrossModalMapper(method="cca", n_components=10)
    >>> mapper.fit(fmri_features, eeg_features)
    >>> fmri_emb, eeg_emb = mapper.transform(fmri_test, eeg_test)
    """

    def __init__(self, method: Literal["cca", "deep"] = "cca", **kwargs) -> None:
        self.method = method

        if method == "cca":
            self._mapper: CCAMapper | DeepCrossModalMapper = CCAMapper(**kwargs)
        elif method == "deep":
            self._mapper = DeepCrossModalMapper(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'cca' or 'deep'.")

    def fit(self, X_fmri: np.ndarray, X_eeg: np.ndarray) -> "CrossModalMapper":
        """Fit the cross-modal mapper."""
        self._mapper.fit(X_fmri, X_eeg)
        return self

    def transform(
        self, X_fmri: np.ndarray, X_eeg: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project both modalities into the shared space.

        Returns
        -------
        (fmri_embedding, eeg_embedding)
        """
        if self.method == "cca":
            return self._mapper.transform(X_fmri, X_eeg)
        else:
            return self._mapper.encode_fmri(X_fmri), self._mapper.encode_eeg(X_eeg)

    @property
    def mapper(self) -> CCAMapper | DeepCrossModalMapper:
        """Access the underlying mapper."""
        return self._mapper
