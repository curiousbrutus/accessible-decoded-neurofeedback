"""
tests/test_fnirs_decoder.py
===========================
Unit tests for the fNIRS decoding module.
"""

import numpy as np
import pytest

from models.fnirs_decoder import FNIRSDecoder, FNIRSFeatureExtractor


@pytest.fixture
def fnirs_data():
    rng = np.random.default_rng(2)
    sfreq = 10.0
    n_ep, n_ch, n_t = 30, 3, int(sfreq * 20)
    hbo = rng.normal(scale=1e-6, size=(n_ep, n_ch, n_t))
    hbr = rng.normal(scale=1e-6, size=(n_ep, n_ch, n_t))
    y = rng.integers(0, 2, size=n_ep)
    hbo[y == 1, 0, 50:120] += 2e-6
    return hbo, hbr, y, sfreq


class TestFNIRSFeatureExtractor:
    def test_output_shape(self, fnirs_data):
        hbo, hbr, _, sfreq = fnirs_data
        extractor = FNIRSFeatureExtractor(sfreq=sfreq)
        features = extractor.transform(hbo, hbr)
        n_ch = hbo.shape[1]
        assert features.shape == (len(hbo), n_ch * 6)  # 6 features per channel

    def test_mismatched_shapes_raises(self, fnirs_data):
        hbo, hbr, _, sfreq = fnirs_data
        extractor = FNIRSFeatureExtractor(sfreq=sfreq)
        with pytest.raises(ValueError):
            extractor.transform(hbo, hbr[:, :2, :])


class TestFNIRSDecoder:
    def test_fit_score(self, fnirs_data):
        hbo, hbr, y, sfreq = fnirs_data
        decoder = FNIRSDecoder(sfreq=sfreq)
        decoder.fit(hbo, hbr, y)
        score = decoder.score(hbo, hbr, y)
        assert 0.0 <= score <= 1.0

    def test_predict_shape(self, fnirs_data):
        hbo, hbr, y, sfreq = fnirs_data
        decoder = FNIRSDecoder(sfreq=sfreq)
        decoder.fit(hbo, hbr, y)
        preds = decoder.predict(hbo, hbr)
        assert preds.shape == (len(hbo),)

    def test_unfitted_raises(self, fnirs_data):
        hbo, hbr, _, sfreq = fnirs_data
        decoder = FNIRSDecoder(sfreq=sfreq)
        with pytest.raises(RuntimeError):
            decoder.predict(hbo, hbr)
