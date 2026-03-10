"""
models
======
Neural decoder modules for fMRI, EEG, and fNIRS signals.

Sub-packages
------------
fmri_decoder    : MVPA and deep learning decoders for fMRI
eeg_decoder     : Feature extraction and classification for EEG
fnirs_decoder   : Feature extraction and classification for fNIRS
"""

from models.fmri_decoder import FMRIDecoder
from models.eeg_decoder import EEGDecoder
from models.fnirs_decoder import FNIRSDecoder

__all__ = ["FMRIDecoder", "EEGDecoder", "FNIRSDecoder"]
