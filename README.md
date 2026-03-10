# Accessible Decoded Neurofeedback

> **Toward scalable closed-loop neurofeedback: integrating fMRI decoding with EEG and fNIRS**

An open-source Python framework for multimodal neurofeedback research, combining
decoded neurofeedback (DecNef), cross-modal neural representation mapping, and
reinforcement learning for adaptive closed-loop experiments.

---

## Overview

Decoded Neurofeedback (DecNef) enables implicit modulation of cognition by
reinforcing specific neural representations decoded from brain imaging data —
but has historically been restricted to expensive, immobile fMRI systems.

This framework investigates whether representations decoded from fMRI can be
approximated using portable EEG and fNIRS modalities, enabling scalable
neurofeedback outside the MRI scanner.

### Research Questions

| RQ  | Question |
|-----|----------|
| RQ1 | Can fMRI-decoded neural representations be predicted from EEG/fNIRS? |
| RQ2 | How can reinforcement learning optimise closed-loop neurofeedback? |
| RQ3 | Does decoded neurofeedback modulate perceptual confidence and metacognition? |
| RQ4 | Can multimodal systems match fMRI decoding accuracy with accessible hardware? |

---

## Repository Structure

```
accessible-decoded-neurofeedback/
├── models/
│   ├── fmri_decoder/          # MVPA (SVM/LogReg) + deep learning (MLP) fMRI decoders
│   ├── eeg_decoder/           # Band-power feature extraction + EEG classifier
│   └── fnirs_decoder/         # HbO/HbR feature extraction + fNIRS classifier
├── cross_modal/
│   └── representation_mapping/  # CCA mapper + contrastive deep cross-modal encoder
├── closed_loop/
│   └── reinforcement_learning_feedback/  # REINFORCE + Actor-Critic RL agents
├── pipelines/
│   └── realtime_neurofeedback/   # End-to-end closed-loop neurofeedback pipeline
├── experiments/
│   ├── perception_task/       # 2AFC visual discrimination paradigm
│   └── confidence_task/       # Metacognitive confidence task (meta-d')
├── notebooks/
│   └── exploratory_analysis/  # Jupyter notebooks (simulation-ready, no hardware needed)
├── docs/
│   ├── project_vision.md      # Research vision and architecture overview
│   └── literature_map.md      # Key references by research theme
└── data/
    └── open_datasets_links.md # Curated open neuroimaging datasets
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/curiousbrutus/accessible-decoded-neurofeedback.git
cd accessible-decoded-neurofeedback
pip install -r requirements.txt
pip install -e .
```

### fMRI Decoding (MVPA)

```python
import numpy as np
from models.fmri_decoder import FMRIDecoder

# Simulated fMRI voxel patterns
X_train = np.random.randn(160, 500)
y_train = np.random.randint(0, 2, 160)

decoder = FMRIDecoder(backend="mvpa", estimator="svm", C=1.0)
decoder.fit(X_train, y_train)
print(decoder.score(X_train, y_train))
```

### Cross-Modal Representation Mapping

```python
from cross_modal import CrossModalMapper

mapper = CrossModalMapper(method="cca", n_components=10)
mapper.fit(fmri_features, eeg_features)
fmri_canon, eeg_canon = mapper.transform(fmri_features, eeg_features)
```

### Closed-Loop RL Neurofeedback

```python
from closed_loop.reinforcement_learning_feedback import NeurofeedbackEnv, ActorCriticAgent

env = NeurofeedbackEnv(n_steps=20, target_similarity=0.7)
agent = ActorCriticAgent(obs_dim=2, n_actions=5)
agent.train(env, n_episodes=500)
```

### Full Pipeline (Simulation Mode)

```python
from pipelines import RealtimeNeurofeedbackPipeline
from models.fmri_decoder import FMRIDecoder
from closed_loop.reinforcement_learning_feedback import ActorCriticAgent

pipeline = RealtimeNeurofeedbackPipeline(
    decoder=FMRIDecoder(backend="mvpa"),
    agent=ActorCriticAgent(),
    verbose=True,
)
pipeline.train_decoder(X_localiser, y_localiser)
pipeline.pretrain_agent(n_episodes=300)
results = pipeline.run_simulation(n_trials=20)
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [`notebooks/exploratory_analysis/01_intro_analysis.ipynb`](notebooks/exploratory_analysis/01_intro_analysis.ipynb) | End-to-end walkthrough: decoding, cross-modal mapping, RL, and experiments |

All notebooks run in **simulation mode** — no neuroimaging hardware required.

---

## Modules

### `models/fmri_decoder`
- `MVPADecoder`: scikit-learn SVM / logistic regression with cross-validation
- `DeepFMRIDecoder`: PyTorch MLP with training loop and model persistence
- `FMRIDecoder`: unified interface switching between MVPA and deep backends

### `models/eeg_decoder`
- `EEGFeatureExtractor`: Welch PSD band-power features (δ, θ, α, β, γ)
- `EEGDecoder`: feature extraction + classification pipeline

### `models/fnirs_decoder`
- `FNIRSFeatureExtractor`: mean / slope / peak features from HbO and HbR
- `FNIRSDecoder`: feature extraction + classification pipeline

### `cross_modal/representation_mapping`
- `CCAMapper`: canonical correlation analysis for linear cross-modal alignment
- `DeepCrossModalMapper`: contrastive (NT-Xent) deep encoder pair
- `CrossModalMapper`: unified interface

### `closed_loop/reinforcement_learning_feedback`
- `NeurofeedbackEnv`: Gymnasium-compatible simulation environment
- `RLFeedbackAgent`: REINFORCE policy-gradient agent
- `ActorCriticAgent`: A2C-style agent with entropy regularisation

### `pipelines/realtime_neurofeedback`
- `RealtimeNeurofeedbackPipeline`: integrates decoder + agent for online sessions

### `experiments`
- `PerceptionTask`: 2AFC visual discrimination with optional neurofeedback
- `ConfidenceTask`: metacognitive confidence task with meta-d' computation

---

## Documentation

- [Project Vision](docs/project_vision.md)
- [Literature Map](docs/literature_map.md)
- [Open Datasets](data/open_datasets_links.md)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch | Deep learning decoders and RL agents |
| scikit-learn | MVPA, CCA, preprocessing |
| MNE | EEG/MEG data processing |
| nilearn | fMRI decoding and masking |
| NumPy / SciPy | Numerical computing |
| Matplotlib / Seaborn | Visualisation |

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

Eyyüb Güven
