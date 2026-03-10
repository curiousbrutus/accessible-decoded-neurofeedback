"""
eeg_features.py
===============
EEG feature extraction and decoding pipeline.

Provides band-power features (delta, theta, alpha, beta, gamma) computed from
epoched EEG data via Welch's power spectral density estimate, along with a
scikit-learn compatible classifier pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Standard EEG frequency bands (Hz)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


class EEGFeatureExtractor:
    """Extract spectral power features from epoched EEG data.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the EEG recording (Hz).
    bands : dict or None
        Frequency bands to compute. Keys are band names and values are
        ``(low_hz, high_hz)`` tuples. Defaults to the standard five bands.
    nperseg : int
        Length of each Welch segment.

    Examples
    --------
    >>> extractor = EEGFeatureExtractor(sfreq=256.0)
    >>> features = extractor.transform(epochs)  # shape: (n_epochs, n_channels * n_bands)
    """

    def __init__(
        self,
        sfreq: float = 256.0,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        nperseg: int = 256,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands or FREQUENCY_BANDS
        self.nperseg = nperseg

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """Extract band-power features from EEG epochs.

        Parameters
        ----------
        epochs : np.ndarray, shape (n_epochs, n_channels, n_times)
            EEG epochs.

        Returns
        -------
        np.ndarray, shape (n_epochs, n_channels * n_bands)
            Log band-power features.
        """
        n_epochs, n_channels, _ = epochs.shape
        n_bands = len(self.bands)
        features = np.zeros((n_epochs, n_channels * n_bands))

        for i, epoch in enumerate(epochs):
            band_powers = self._compute_band_powers(epoch)
            features[i] = band_powers.ravel()

        return features

    def _compute_band_powers(self, epoch: np.ndarray) -> np.ndarray:
        """Compute log band power for one epoch.

        Parameters
        ----------
        epoch : np.ndarray, shape (n_channels, n_times)

        Returns
        -------
        np.ndarray, shape (n_channels, n_bands)
        """
        n_channels = epoch.shape[0]
        n_bands = len(self.bands)
        band_powers = np.zeros((n_channels, n_bands))

        for ch_idx in range(n_channels):
            freqs, psd = welch(epoch[ch_idx], fs=self.sfreq, nperseg=self.nperseg)
            for band_idx, (low, high) in enumerate(self.bands.values()):
                mask = (freqs >= low) & (freqs < high)
                power = np.mean(psd[mask]) if mask.any() else 0.0
                band_powers[ch_idx, band_idx] = np.log1p(power)

        return band_powers

    def feature_names(self, n_channels: int) -> List[str]:
        """List of feature names in the order they appear in the output.

        Parameters
        ----------
        n_channels : int
            Number of EEG channels.

        Returns
        -------
        list[str]
        """
        return [f"ch{ch}_{band}" for ch in range(n_channels) for band in self.bands]


class EEGDecoder(BaseEstimator, ClassifierMixin):
    """End-to-end EEG decoder: feature extraction + classification.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    bands : dict or None
        Frequency bands for power spectrum features.
    estimator : str
        ``"logreg"`` (default) or ``"svm"``.
    C : float
        Regularisation parameter.
    random_state : int or None
        Random seed.

    Examples
    --------
    >>> decoder = EEGDecoder(sfreq=256.0)
    >>> decoder.fit(epochs_train, y_train)
    >>> acc = decoder.score(epochs_test, y_test)
    """

    def __init__(
        self,
        sfreq: float = 256.0,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        estimator: str = "logreg",
        C: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands
        self.estimator = estimator
        self.C = C
        self.random_state = random_state

        self._extractor: Optional[EEGFeatureExtractor] = None
        self._pipeline: Optional[Pipeline] = None

    def fit(self, epochs: np.ndarray, y: np.ndarray) -> "EEGDecoder":
        """Fit the EEG decoder on training epochs.

        Parameters
        ----------
        epochs : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,)

        Returns
        -------
        self
        """
        self._extractor = EEGFeatureExtractor(sfreq=self.sfreq, bands=self.bands)
        X = self._extractor.transform(epochs)

        if self.estimator == "logreg":
            clf = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        else:
            clf = SVC(C=self.C, kernel="linear", probability=True, random_state=self.random_state)

        self._pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self._pipeline.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, epochs: np.ndarray) -> np.ndarray:
        """Predict class labels for new epochs."""
        X = self._get_features(epochs)
        return self._pipeline.predict(X)

    def predict_proba(self, epochs: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        X = self._get_features(epochs)
        return self._pipeline.predict_proba(X)

    def score(self, epochs: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy."""
        X = self._get_features(epochs)
        return self._pipeline.score(X, y)

    def _get_features(self, epochs: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._extractor.transform(epochs)

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("EEGDecoder has not been fitted. Call fit() first.")
