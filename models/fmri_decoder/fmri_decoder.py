"""
fmri_decoder.py
===============
Unified FMRIDecoder interface that wraps both MVPA and deep learning backends.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from models.fmri_decoder.deep_decoder import DeepFMRIDecoder
from models.fmri_decoder.mvpa_decoder import MVPADecoder


class FMRIDecoder:
    """Unified fMRI decoder supporting both MVPA and deep learning backends.

    Parameters
    ----------
    backend : str
        Decoding backend: ``"mvpa"`` (default) or ``"deep"``.
    **kwargs
        Additional keyword arguments forwarded to the underlying decoder.

    Examples
    --------
    >>> decoder = FMRIDecoder(backend="mvpa", estimator="svm", C=1.0)
    >>> decoder.fit(X_train, y_train)
    >>> print(decoder.score(X_test, y_test))

    >>> decoder = FMRIDecoder(backend="deep", n_voxels=5000, n_classes=2)
    >>> decoder.fit(X_train, y_train)
    """

    def __init__(
        self,
        backend: Literal["mvpa", "deep"] = "mvpa",
        **kwargs,
    ) -> None:
        self.backend = backend

        if backend == "mvpa":
            self._decoder: MVPADecoder | DeepFMRIDecoder = MVPADecoder(**kwargs)
        elif backend == "deep":
            self._decoder = DeepFMRIDecoder(**kwargs)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'mvpa' or 'deep'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FMRIDecoder":
        """Fit the decoder on training data."""
        self._decoder.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self._decoder.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        return self._decoder.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return self._decoder.score(X, y)

    @property
    def decoder(self) -> MVPADecoder | DeepFMRIDecoder:
        """Access the underlying decoder instance."""
        return self._decoder
