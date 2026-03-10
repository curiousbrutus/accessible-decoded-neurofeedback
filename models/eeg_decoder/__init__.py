"""
eeg_decoder
===========
EEG feature extraction and classification pipeline.

Classes
-------
EEGFeatureExtractor : Extracts band-power and connectivity features from raw EEG
EEGDecoder          : End-to-end EEG decoding (feature extraction + classification)
"""

from models.eeg_decoder.eeg_features import EEGFeatureExtractor, EEGDecoder

__all__ = ["EEGFeatureExtractor", "EEGDecoder"]
