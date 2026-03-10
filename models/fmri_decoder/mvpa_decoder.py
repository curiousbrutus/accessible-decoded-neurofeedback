"""
mvpa_decoder.py
===============
Multivariate Pattern Analysis (MVPA) decoder for fMRI data.

Wraps scikit-learn estimators with nilearn masking utilities to provide
a clean interface for training and applying linear and non-linear classifiers
to fMRI volumetric data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MVPADecoder(BaseEstimator, ClassifierMixin):
    """MVPA decoder for fMRI patterns.

    Trains a multivariate pattern classifier on pre-masked fMRI voxel
    patterns. Supports any scikit-learn compatible estimator.

    Parameters
    ----------
    estimator : str
        One of ``"svm"`` (default), ``"logreg"``, or a custom
        scikit-learn estimator instance.
    C : float
        Regularisation strength (lower = stronger regularisation).
    kernel : str
        Kernel type for SVM (``"linear"``, ``"rbf"``, etc.).
    scale : bool
        Whether to z-score voxel patterns before classification.
    random_state : int or None
        Random seed for reproducibility.

    Examples
    --------
    >>> decoder = MVPADecoder(estimator="svm", C=1.0)
    >>> decoder.fit(X_train, y_train)
    >>> accuracy = decoder.score(X_test, y_test)
    """

    def __init__(
        self,
        estimator: str = "svm",
        C: float = 1.0,
        kernel: str = "linear",
        scale: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        self.estimator = estimator
        self.C = C
        self.kernel = kernel
        self.scale = scale
        self.random_state = random_state

        self._pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        steps: list = []
        if self.scale:
            steps.append(("scaler", StandardScaler()))

        if self.estimator == "svm":
            clf = SVC(
                C=self.C,
                kernel=self.kernel,
                probability=True,
                random_state=self.random_state,
            )
        elif self.estimator == "logreg":
            clf = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
            )
        else:
            clf = self.estimator

        steps.append(("classifier", clf))
        return Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MVPADecoder":
        """Fit the MVPA decoder.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
            fMRI voxel patterns (one row per trial / TR).
        y : np.ndarray, shape (n_samples,)
            Integer class labels for each pattern.

        Returns
        -------
        self
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new patterns.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        self._check_fitted()
        return self._pipeline.predict(X)

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
        return self._pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return mean accuracy on the given test patterns.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        float
        """
        self._check_fitted()
        return self._pipeline.score(X, y)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> np.ndarray:
        """Run k-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
        y : np.ndarray, shape (n_samples,)
        cv : int
            Number of cross-validation folds.
        scoring : str
            Scoring metric.

        Returns
        -------
        np.ndarray, shape (cv,)
            Per-fold scores.
        """
        pipeline = self._build_pipeline()
        return cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Decoder has not been fitted yet. Call fit() first.")
