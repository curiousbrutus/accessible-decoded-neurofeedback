"""
tests/test_cross_modal.py
=========================
Unit tests for the cross-modal representation mapping module.
"""

import numpy as np
import pytest

from cross_modal import CrossModalMapper
from cross_modal.representation_mapping.cross_modal_model import CCAMapper, DeepCrossModalMapper


@pytest.fixture
def paired_data():
    rng = np.random.default_rng(3)
    n = 80
    fmri = rng.normal(size=(n, 20))
    eeg = rng.normal(size=(n, 15))
    # Add shared latent structure
    latent = rng.normal(size=(n, 5))
    fmri[:, :5] += 2.0 * latent
    eeg[:, :5] += 2.0 * latent
    return fmri, eeg


class TestCCAMapper:
    def test_fit_transform(self, paired_data):
        fmri, eeg = paired_data
        mapper = CCAMapper(n_components=3)
        mapper.fit(fmri, eeg)
        fmri_c, eeg_c = mapper.transform(fmri, eeg)
        assert fmri_c.shape == (len(fmri), 3)
        assert eeg_c.shape == (len(eeg), 3)

    def test_correlations_positive(self, paired_data):
        fmri, eeg = paired_data
        mapper = CCAMapper(n_components=3)
        mapper.fit(fmri, eeg)
        fmri_c, eeg_c = mapper.transform(fmri, eeg)
        for i in range(3):
            corr = np.corrcoef(fmri_c[:, i], eeg_c[:, i])[0, 1]
            assert corr > 0.0  # shared structure should produce positive correlations

    def test_unfitted_raises(self, paired_data):
        fmri, eeg = paired_data
        mapper = CCAMapper()
        with pytest.raises(RuntimeError):
            mapper.transform(fmri, eeg)


class TestDeepCrossModalMapper:
    def test_fit_encode(self, paired_data):
        fmri, eeg = paired_data
        mapper = DeepCrossModalMapper(
            n_fmri_features=20,
            n_eeg_features=15,
            embed_dim=8,
            epochs=2,
            batch_size=16,
        )
        mapper.fit(fmri, eeg)
        fmri_emb = mapper.encode_fmri(fmri)
        eeg_emb = mapper.encode_eeg(eeg)
        assert fmri_emb.shape == (len(fmri), 8)
        assert eeg_emb.shape == (len(eeg), 8)
        # Embeddings should be unit-normalised
        norms = np.linalg.norm(fmri_emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_unfitted_raises(self, paired_data):
        fmri, _ = paired_data
        mapper = DeepCrossModalMapper(n_fmri_features=20, n_eeg_features=15)
        with pytest.raises(RuntimeError):
            mapper.encode_fmri(fmri)


class TestCrossModalMapper:
    def test_cca_method(self, paired_data):
        fmri, eeg = paired_data
        mapper = CrossModalMapper(method="cca", n_components=4)
        mapper.fit(fmri, eeg)
        fmri_c, eeg_c = mapper.transform(fmri, eeg)
        assert fmri_c.shape == (len(fmri), 4)
        assert eeg_c.shape == (len(eeg), 4)

    def test_deep_method(self, paired_data):
        fmri, eeg = paired_data
        mapper = CrossModalMapper(
            method="deep",
            n_fmri_features=20,
            n_eeg_features=15,
            embed_dim=6,
            epochs=2,
            batch_size=16,
        )
        mapper.fit(fmri, eeg)
        fmri_emb, eeg_emb = mapper.transform(fmri, eeg)
        assert fmri_emb.shape == (len(fmri), 6)
        assert eeg_emb.shape == (len(eeg), 6)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            CrossModalMapper(method="invalid")
