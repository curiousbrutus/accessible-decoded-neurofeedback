"""
fnirs_features.py
=================
fNIRS feature extraction and decoding pipeline.

Computes haemodynamic features (mean HbO/HbR, slope, peak) from fNIRS
channel timeseries, suitable for neural state classification and as inputs
to the cross-modal representation mapping module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class FNIRSFeatureExtractor:
    """Extract haemodynamic features from fNIRS epoch data.

    For each channel the following features are computed:
    * Mean HbO and HbR amplitude
    * Slope of the HbO/HbR response (linear regression over time)
    * Peak amplitude of HbO and HbR

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the fNIRS recording (Hz).
    tmin : float
        Start of the feature extraction window relative to epoch onset (s).
    tmax : float
        End of the feature extraction window relative to epoch onset (s).

    Examples
    --------
    >>> extractor = FNIRSFeatureExtractor(sfreq=10.0, tmin=5.0, tmax=15.0)
    >>> features = extractor.transform(epochs_hbo, epochs_hbr)
    """

    # Number of features computed per channel pair (HbO + HbR together)
    N_FEATURES_PER_CHANNEL = 6  # mean_hbo, slope_hbo, peak_hbo, mean_hbr, slope_hbr, peak_hbr

    def __init__(
        self,
        sfreq: float = 10.0,
        tmin: float = 0.0,
        tmax: float = 20.0,
    ) -> None:
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax

    def transform(
        self,
        epochs_hbo: np.ndarray,
        epochs_hbr: np.ndarray,
    ) -> np.ndarray:
        """Extract features from HbO and HbR epochs.

        Parameters
        ----------
        epochs_hbo : np.ndarray, shape (n_epochs, n_channels, n_times)
            Oxyhaemoglobin concentration changes (HbO).
        epochs_hbr : np.ndarray, shape (n_epochs, n_channels, n_times)
            Deoxyhaemoglobin concentration changes (HbR).

        Returns
        -------
        np.ndarray, shape (n_epochs, n_channels * N_FEATURES_PER_CHANNEL)
        """
        if epochs_hbo.shape != epochs_hbr.shape:
            raise ValueError(
                f"HbO and HbR epoch arrays must have the same shape; "
                f"got {epochs_hbo.shape} and {epochs_hbr.shape}."
            )

        n_epochs, n_channels, n_times = epochs_hbo.shape

        # Determine sample indices for extraction window
        start = int(self.tmin * self.sfreq)
        stop = int(self.tmax * self.sfreq)
        stop = min(stop, n_times)
        time_vec = np.arange(stop - start) / self.sfreq

        features = np.zeros((n_epochs, n_channels * self.N_FEATURES_PER_CHANNEL))

        for i in range(n_epochs):
            row = []
            for ch in range(n_channels):
                hbo = epochs_hbo[i, ch, start:stop]
                hbr = epochs_hbr[i, ch, start:stop]
                row.extend(self._channel_features(hbo, time_vec))
                row.extend(self._channel_features(hbr, time_vec))
            features[i] = row

        return features

    @staticmethod
    def _channel_features(signal: np.ndarray, time_vec: np.ndarray) -> List[float]:
        """Compute [mean, slope, peak] for a single channel signal."""
        mean_val = float(np.mean(signal))
        peak_val = float(np.max(np.abs(signal)))

        # Least-squares slope
        if len(signal) > 1:
            coeffs = np.polyfit(time_vec, signal, 1)
            slope = float(coeffs[0])
        else:
            slope = 0.0

        return [mean_val, slope, peak_val]


class FNIRSDecoder(BaseEstimator, ClassifierMixin):
    """End-to-end fNIRS decoder.

    Parameters
    ----------
    sfreq : float
        Sampling frequency (Hz).
    tmin : float
        Start of the haemodynamic response window (s).
    tmax : float
        End of the haemodynamic response window (s).
    estimator : str
        ``"logreg"`` (default) or ``"svm"``.
    C : float
        Regularisation strength.
    random_state : int or None
        Random seed.

    Examples
    --------
    >>> decoder = FNIRSDecoder(sfreq=10.0, tmin=5.0, tmax=15.0)
    >>> decoder.fit(hbo_train, hbr_train, y_train)
    >>> acc = decoder.score(hbo_test, hbr_test, y_test)
    """

    def __init__(
        self,
        sfreq: float = 10.0,
        tmin: float = 0.0,
        tmax: float = 20.0,
        estimator: str = "logreg",
        C: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.estimator = estimator
        self.C = C
        self.random_state = random_state

        self._extractor: Optional[FNIRSFeatureExtractor] = None
        self._pipeline: Optional[Pipeline] = None

    def fit(
        self,
        epochs_hbo: np.ndarray,
        epochs_hbr: np.ndarray,
        y: np.ndarray,
    ) -> "FNIRSDecoder":
        """Fit the fNIRS decoder.

        Parameters
        ----------
        epochs_hbo : np.ndarray, shape (n_epochs, n_channels, n_times)
        epochs_hbr : np.ndarray, shape (n_epochs, n_channels, n_times)
        y : np.ndarray, shape (n_epochs,)

        Returns
        -------
        self
        """
        self._extractor = FNIRSFeatureExtractor(
            sfreq=self.sfreq, tmin=self.tmin, tmax=self.tmax
        )
        X = self._extractor.transform(epochs_hbo, epochs_hbr)

        if self.estimator == "logreg":
            clf = LogisticRegression(C=self.C, max_iter=1000, random_state=self.random_state)
        else:
            clf = SVC(C=self.C, kernel="linear", probability=True, random_state=self.random_state)

        self._pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self._pipeline.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, epochs_hbo: np.ndarray, epochs_hbr: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = self._get_features(epochs_hbo, epochs_hbr)
        return self._pipeline.predict(X)

    def predict_proba(self, epochs_hbo: np.ndarray, epochs_hbr: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        X = self._get_features(epochs_hbo, epochs_hbr)
        return self._pipeline.predict_proba(X)

    def score(
        self,
        epochs_hbo: np.ndarray,
        epochs_hbr: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Return mean classification accuracy."""
        X = self._get_features(epochs_hbo, epochs_hbr)
        return self._pipeline.score(X, y)

    def _get_features(self, epochs_hbo: np.ndarray, epochs_hbr: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._extractor.transform(epochs_hbo, epochs_hbr)

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("FNIRSDecoder has not been fitted. Call fit() first.")
