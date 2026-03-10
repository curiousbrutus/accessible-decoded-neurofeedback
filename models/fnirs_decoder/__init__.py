"""
fnirs_decoder
=============
fNIRS feature extraction and classification pipeline.

Classes
-------
FNIRSFeatureExtractor : Extracts HbO/HbR features from fNIRS channels
FNIRSDecoder          : End-to-end fNIRS decoder
"""

from models.fnirs_decoder.fnirs_features import FNIRSFeatureExtractor, FNIRSDecoder

__all__ = ["FNIRSFeatureExtractor", "FNIRSDecoder"]
