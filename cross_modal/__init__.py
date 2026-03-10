"""
cross_modal
===========
Cross-modal neural representation mapping between fMRI and EEG/fNIRS.

Sub-packages
------------
representation_mapping : Models that map fMRI representations to EEG/fNIRS features
"""

from cross_modal.representation_mapping import CrossModalMapper

__all__ = ["CrossModalMapper"]
