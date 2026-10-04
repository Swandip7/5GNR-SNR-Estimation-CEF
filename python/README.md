
`train_cef.py` runs the full experimental pipeline.


## Experimental protocol

- **Leave-one-CDL-out**: for each of four experiments, three CDL profiles form the pretraining pool and the fourth is held out.
- **Fine-tuning budget**: 1,500 samples from the held-out profile, validated on a separate 10% split, tested on a 50% split never used for model selection.
- **Baselines**: MLP-16 (feature-only, no classical estimates), ResNet1D, Dilated-CNN, XGBoost. All receive identical inputs to CEF except MLP-16, which is by construction feature-only.
- **Ablations**: 2x2 fusion x scenario-conditioning grid, and a per-estimator drop-one study.
- **OOD evaluation**: 5,000-sample set with severity-shifted impairment parameters.

## CEF model

- Input: 16 receiver-observable features concatenated with 4 classical SNR estimates (LS, ML, EVM, DD), 20-dimensional total.
- Architecture: 5-layer MLP - `20 -> 256 -> 128 -> 64 -> 32 -> 1`, GELU activations, BatchNorm before the third activation.
- Parameters: 49,793.
- Loss: Huber (delta = 1.5 in standardized label units).
- Two-stage training: pretrain on 3 CDLs (60 epochs), fine-tune on held-out CDL (20 epochs).

## Runtime

Approximately 45-70 minutes on an NVIDIA T4 GPU for the full pipeline, including sensitivity sweep, cross-CDL experiments, ablations, and OOD evaluation.

