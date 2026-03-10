"""
tests/test_fmri_decoder.py
===========================
Unit tests for the fMRI decoding module.
"""

import numpy as np
import pytest

from models.fmri_decoder import FMRIDecoder, MVPADecoder, DeepFMRIDecoder


@pytest.fixture
def fmri_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 100))
    y = rng.integers(0, 2, size=60)
    # Add discriminative signal
    X[y == 1, :10] += 1.5
    return X, y


class TestMVPADecoder:
    def test_fit_predict(self, fmri_data):
        X, y = fmri_data
        decoder = MVPADecoder(estimator="svm", C=1.0)
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba(self, fmri_data):
        X, y = fmri_data
        decoder = MVPADecoder(estimator="svm")
        decoder.fit(X, y)
        probs = decoder.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_score(self, fmri_data):
        X, y = fmri_data
        decoder = MVPADecoder(estimator="logreg")
        decoder.fit(X, y)
        score = decoder.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_cross_validate(self, fmri_data):
        X, y = fmri_data
        decoder = MVPADecoder()
        cv_scores = decoder.cross_validate(X, y, cv=3)
        assert cv_scores.shape == (3,)

    def test_unfitted_raises(self):
        decoder = MVPADecoder()
        with pytest.raises(RuntimeError):
            decoder.predict(np.zeros((5, 100)))


class TestDeepFMRIDecoder:
    def test_fit_predict(self, fmri_data):
        X, y = fmri_data
        decoder = DeepFMRIDecoder(n_voxels=100, n_classes=2, epochs=3, batch_size=16)
        decoder.fit(X, y)
        preds = decoder.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba(self, fmri_data):
        X, y = fmri_data
        decoder = DeepFMRIDecoder(n_voxels=100, n_classes=2, epochs=3, batch_size=16)
        decoder.fit(X, y)
        probs = decoder.predict_proba(X)
        assert probs.shape == (len(X), 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_loss_history(self, fmri_data):
        X, y = fmri_data
        decoder = DeepFMRIDecoder(n_voxels=100, n_classes=2, epochs=3)
        decoder.fit(X, y)
        assert len(decoder.loss_history) == 3

    def test_unfitted_raises(self):
        decoder = DeepFMRIDecoder(n_voxels=100, n_classes=2)
        with pytest.raises(RuntimeError):
            decoder.predict(np.zeros((5, 100)))


class TestFMRIDecoder:
    def test_mvpa_backend(self, fmri_data):
        X, y = fmri_data
        decoder = FMRIDecoder(backend="mvpa", estimator="svm")
        decoder.fit(X, y)
        assert decoder.score(X, y) >= 0.0

    def test_deep_backend(self, fmri_data):
        X, y = fmri_data
        decoder = FMRIDecoder(backend="deep", n_voxels=100, n_classes=2, epochs=3)
        decoder.fit(X, y)
        assert decoder.score(X, y) >= 0.0

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            FMRIDecoder(backend="invalid")
