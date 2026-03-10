# Project Vision

## Toward Accessible Closed-Loop Neurofeedback

**Accessible Decoded Neurofeedback** is an open-source research framework that
bridges high-precision decoded neurofeedback (DecNef) — historically restricted
to fMRI — with the portable, low-cost modalities of EEG and fNIRS.

---

## Motivation

Recent advances in multivariate pattern analysis and deep learning have made
it possible to decode fine-grained neural representations from fMRI data and
to use those decoded states as feedback signals that modulate cognition without
participants' explicit awareness (DecNef).

However, fMRI-based neurofeedback is expensive, immobile, and difficult to
scale — limiting its translation into clinical and everyday settings.

EEG and fNIRS are affordable, portable, and increasingly reliable, but their
spatial resolution is insufficient for high-dimensional representation decoding
when used in isolation.

**The central hypothesis of this project:**
> If neural representations can be decoded with precision using fMRI, can those
> same representations be *approximated* or *tracked* using EEG and fNIRS
> through learned cross-modal mappings?

---

## Long-Term Vision

1. **Cross-modal decoding**: Develop computational bridges that map
   fMRI-level representations onto EEG/fNIRS feature spaces using contrastive
   learning, canonical correlation analysis, and deep transformers.

2. **Adaptive closed-loop systems**: Apply reinforcement learning to
   continuously optimise feedback signals based on both neural and behavioural
   outcomes during a session.

3. **Accessible neurotechnology**: Enable decoded neurofeedback to operate
   *outside the MRI scanner* using portable neuroimaging, broadening access for
   research and potential therapeutic applications.

4. **Open science**: All pipelines, models, and experimental paradigms are
   released as open-source software to support the wider neuroscience community.

---

## Core Research Questions

| RQ  | Question |
|-----|----------|
| RQ1 | Can fMRI-decoded representations be approximated from EEG/fNIRS? |
| RQ2 | How can RL optimise closed-loop neurofeedback for cognitive training? |
| RQ3 | Does decoded neurofeedback modulate perceptual confidence and metacognition? |
| RQ4 | Can multimodal systems match fMRI decoding performance with accessible hardware? |

---

## Framework Architecture

```
accessible-decoded-neurofeedback/
├── models/
│   ├── fmri_decoder/         # MVPA + deep learning decoders
│   ├── eeg_decoder/          # Band-power feature extraction + classification
│   └── fnirs_decoder/        # Haemodynamic feature extraction + classification
├── cross_modal/
│   └── representation_mapping/  # CCA + contrastive deep cross-modal models
├── closed_loop/
│   └── reinforcement_learning_feedback/  # REINFORCE + Actor-Critic agents
├── pipelines/
│   └── realtime_neurofeedback/   # End-to-end closed-loop pipeline
├── experiments/
│   ├── perception_task/          # 2AFC visual discrimination
│   └── confidence_task/          # Metacognitive confidence paradigm
├── notebooks/
│   └── exploratory_analysis/     # Interactive Jupyter notebooks
├── docs/                          # Documentation
└── data/                          # Open dataset links and preprocessing scripts
```

---

## Guiding Principles

- **Modularity**: Each component is independently replaceable.
- **Reproducibility**: All experiments use deterministic seeds and documented pipelines.
- **Accessibility**: Code runs in simulation mode without neuroimaging hardware.
- **Open science**: MIT-licensed, with full documentation.
