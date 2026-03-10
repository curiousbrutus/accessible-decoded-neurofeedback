# Open Datasets

A curated list of publicly available neuroimaging datasets relevant to the
Accessible Decoded Neurofeedback project.

---

## fMRI Datasets

### 1. Haxby 2001 Object Perception (OpenNeuro ds000105)
- **URL**: https://openneuro.org/datasets/ds000105
- **Description**: Whole-brain fMRI while viewing 8 categories of objects
  (faces, cats, houses, chairs, scissors, shoes, bottles, scrambled images).
- **Relevance**: Classic MVPA benchmark; ideal for validating fMRI decoders.
- **Accessible via**: `nilearn.datasets.fetch_haxby()`

### 2. Natural Scenes Dataset (NSD)
- **URL**: https://naturalscenesdataset.org/
- **Description**: Ultra-high-resolution 7T fMRI from 8 participants viewing
  73,000 natural image stimuli.
- **Relevance**: Large-scale fMRI decoding and reconstruction benchmark.
- **Note**: Requires registration; ~1 TB per subject.

### 3. Human Connectome Project (HCP)
- **URL**: https://www.humanconnectome.org/study/hcp-young-adult
- **Description**: High-resolution 3T fMRI, 1200 subjects, multiple tasks.
- **Relevance**: Cross-subject generalisation; resting-state connectivity.
- **Accessible via**: `nilearn.datasets.fetch_hcp_rest()`

### 4. Studyforrest
- **URL**: https://www.studyforrest.org/
- **Description**: 7T fMRI while watching Forrest Gump; naturalistic paradigm.
- **Relevance**: Naturalistic neural decoding; long continuous recordings.

---

## EEG Datasets

### 5. MOABB (Mother of All BCI Benchmarks)
- **URL**: https://moabb.neurotechx.com/
- **Description**: Curated benchmark suite of EEG motor imagery, P300, and SSVEP datasets.
- **Relevance**: EEG decoder validation; standardised comparison protocol.
- **Install**: `pip install moabb`

### 6. EEG-ImageNet (THINGS-EEG)
- **URL**: https://osf.io/3jk45/
- **Description**: EEG responses to 22,248 natural images from THINGS database.
- **Relevance**: EEG-based image decoding; cross-modal mapping with fMRI.

### 7. BCI Competition IV Dataset 2a
- **URL**: https://www.bbci.de/competition/iv/
- **Description**: 22-channel EEG, 4-class motor imagery, 9 subjects.
- **Relevance**: Standard EEG motor decoding benchmark.

---

## fNIRS Datasets

### 8. fNIRS-EEG Motor Imagery Dataset
- **URL**: https://github.com/canlabcmu/fnirs-eeg-motor-imagery
- **Description**: Simultaneous fNIRS and EEG during motor imagery tasks.
- **Relevance**: Multimodal fNIRS + EEG decoding; cross-modal mapping.

### 9. fNIRS Mental Workload Dataset
- **URL**: https://osf.io/tduf8/
- **Description**: fNIRS during N-back tasks across 3 difficulty levels.
- **Relevance**: Cognitive state decoding with fNIRS.

### 10. fNIRS Social Interaction Dataset (Scholkmann et al.)
- **URL**: https://www.nature.com/articles/s41597-023-02114-1
- **Description**: fNIRS recorded during naturalistic social interactions.
- **Relevance**: Ecological validity; naturalistic neurofeedback paradigms.

---

## Multimodal / Combined

### 11. Simultaneous fMRI-EEG Dataset (Dähne et al.)
- **URL**: https://openneuro.org/datasets/ds000117
- **Description**: Simultaneous fMRI and EEG during face/object perception.
- **Relevance**: Ground truth for cross-modal fMRI-EEG representation mapping.

### 12. fMRI + fNIRS Finger Tapping (Naseer et al.)
- **Description**: Concurrent fMRI and fNIRS motor paradigm.
- **Relevance**: Validating fNIRS as a proxy for fMRI BOLD signal.

---

## How to Access

Most datasets are accessible via:

```python
# nilearn (fMRI)
from nilearn import datasets
data = datasets.fetch_haxby()

# MOABB (EEG)
from moabb.datasets import BNCI2014001
dataset = BNCI2014001()
dataset.download()

# MNE datasets (EEG/MEG)
import mne
mne.datasets.sample.data_path()
```

For datasets requiring registration (NSD, HCP), follow the links above and
store downloaded data in a local `data/raw/` directory (excluded from version
control via `.gitignore`).
