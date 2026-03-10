"""
fmri_decoder
============
fMRI neural decoding using multivariate pattern analysis (MVPA) and deep learning.

Classes
-------
MVPADecoder       : Scikit-learn based MVPA decoder (SVM, logistic regression, etc.)
DeepFMRIDecoder   : PyTorch deep learning decoder for fMRI volumes
FMRIDecoder       : Unified interface wrapping both MVPA and deep decoders
"""

from models.fmri_decoder.mvpa_decoder import MVPADecoder
from models.fmri_decoder.deep_decoder import DeepFMRIDecoder
from models.fmri_decoder.fmri_decoder import FMRIDecoder

__all__ = ["MVPADecoder", "DeepFMRIDecoder", "FMRIDecoder"]
