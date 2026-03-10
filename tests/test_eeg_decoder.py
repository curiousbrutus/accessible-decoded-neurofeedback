"""
tests/test_eeg_decoder.py
=========================
Unit tests for the EEG decoding module.
"""

import numpy as np
import pytest

from models.eeg_decoder import EEGDecoder, EEGFeatureExtractor


@pytest.fixture
def eeg_data():
    rng = np.random.default_rng(1)
    sfreq = 128.0
    n_epochs, n_ch, n_t = 40, 4, int(sfreq * 2)
    epochs = rng.normal(size=(n_epochs, n_ch, n_t))
    y = rng.integers(0, 2, size=n_epochs)
    t = np.arange(n_t) / sfreq
    epochs[y == 1, 0, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    return epochs, y, sfreq


class TestEEGFeatureExtractor:
    def test_output_shape(self, eeg_data):
        epochs, _, sfreq = eeg_data
        extractor = EEGFeatureExtractor(sfreq=sfreq, nperseg=64)
        features = extractor.transform(epochs)
        n_bands = 5
        assert features.shape == (len(epochs), epochs.shape[1] * n_bands)

    def test_log_transform(self, eeg_data):
        epochs, _, sfreq = eeg_data
        extractor = EEGFeatureExtractor(sfreq=sfreq, nperseg=64)
        features = extractor.transform(epochs)
        # log1p values should be >= 0
        assert np.all(features >= 0)


class TestEEGDecoder:
    def test_fit_score(self, eeg_data):
        epochs, y, sfreq = eeg_data
        decoder = EEGDecoder(sfreq=sfreq)
        decoder.fit(epochs, y)
        score = decoder.score(epochs, y)
        assert 0.0 <= score <= 1.0

    def test_predict_shape(self, eeg_data):
        epochs, y, sfreq = eeg_data
        decoder = EEGDecoder(sfreq=sfreq)
        decoder.fit(epochs, y)
        preds = decoder.predict(epochs)
        assert preds.shape == (len(epochs),)

    def test_predict_proba_shape(self, eeg_data):
        epochs, y, sfreq = eeg_data
        decoder = EEGDecoder(sfreq=sfreq)
        decoder.fit(epochs, y)
        probs = decoder.predict_proba(epochs)
        assert probs.shape == (len(epochs), 2)

    def test_unfitted_raises(self, eeg_data):
        epochs, _, sfreq = eeg_data
        decoder = EEGDecoder(sfreq=sfreq)
        with pytest.raises(RuntimeError):
            decoder.predict(epochs)
