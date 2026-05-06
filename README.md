# ecg-augmentations

Python library for ECG signal augmentation compatible with PyTorch.
Developed as part of a bachelor's thesis at VUT-FEKT (BUT-FEEC) (2026).

## Installation

```bash
pip install git+https://github.com/ShatovDaniil/ecg-augmentations.git
```

Or clone and install locally:

```bash
git clone https://github.com/ShatovDaniil/ecg-augmentations.git
cd ecg-augmentations
pip install -e .
```

## Requirements

- Python >= 3.8
- torch >= 1.9.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- wfdb >= 3.4.0

## Quick Start

```python
from ecg_augmentations import RandomNoiseAdder, HighPassFilter, SignalDrift

# Add Gaussian noise (recommended range: 0-10%)
transform = RandomNoiseAdder(low_limit=0.0, high_limit=0.05,
                             p=1.0, use_on='sample')
augmented, marks = transform(signal, marks, global_apply_to_all=True)

# High-pass filter (recommended range: 0.1-10 Hz)
hpf = HighPassFilter(critical_f=1.0, sampling_f=500,
                     order=101, p=1.0, use_on='sample')
filtered, marks = hpf(signal, marks, global_apply_to_all=True)

# Linear drift (recommended range: amplitude 0-0.05)
drift = SignalDrift(drift_type='linear', amplitude=0.03,
                    period=1.0, sampling_f=500, p=1.0, use_on='sample')
drifted, marks = drift(signal, marks, global_apply_to_all=True)
```

## Interface

Each augmentation class inherits from `BaseTransform` and uses a unified interface:

```python
augmented_signal, augmented_marks = transform(x, y, global_apply_to_all=True)
```

- `x` — 1D PyTorch tensor (single ECG channel)
- `y` — list of annotations in format `[(start, end), ...]`
- `global_apply_to_all` — if `True`, ignores probability `p` and always applies

### Global parameters

| Parameter | Type | Description |
|---|---|---|
| `use_on` | str | `'sample'` (signal only), `'target'` (annotations only), `'all'` (both) |
| `p` | float | Probability of activation per call (0.0–1.0) |
| `apply_to_all_channels` | bool | Apply to all channels regardless of `p` |

## Available augmentations

| Class | Description | Recommended range | Category |
|---|---|---|---|
| `RandomNoiseAdder` | Gaussian noise | 0–10 % | safe |
| `PowerlineNoise` | 50 Hz powerline interference | 0–0.02 mV | dangerous |
| `BaselineWander` | Low-frequency baseline drift (0.2 Hz) | 0–0.05 mV | medium |
| `EMGNoise` | High-frequency muscle noise (>100 Hz) | 0–0.5 | medium |
| `HighPassFilter` | FIR high-pass filter (Nuttall window) | 0.1–10 Hz | safe |
| `LowPassFilter` | FIR low-pass filter (Nuttall window) | not recommended | dangerous |
| `Scaling` | Global amplitude scaling | factor ±1.0 | medium |
| `RandomAmplitudeVariation` | Local amplitude variation per segment | factor ±1.0 | medium |
| `ClippingDistortion` | Amplitude clipping (amplifier saturation) | no limit | safe |
| `SignalInverter` | Polarity inversion (×−1) | not recommended | dangerous |
| `SignalDrift` | Baseline drift: linear / sinusoidal / random | 0–0.05 (linear/sin) | medium |
| `SinusoidalStretching` | Sinusoidal amplitude modulation | 0–0.5 | medium |
| `EchoAdder` | Echo effect (delayed copy) | not recommended | dangerous |
| `SignalArtefactAdder` | Random impulse artefacts | with caution | medium |
| `ZeroOutSignal` | Random signal dropout | 0–40 % | safe |
| `ZeroOutChannel` | Full channel zeroing | not recommended | critical |
| `SegmentShuffle` | Random segment reordering | not recommended | dangerous |
| `RandomSignalReplacement` | Segment replacement from LUDB | 0–100 samples | medium |
| `SignalTimeStretch` | Time stretching via interpolation | factor 0.9–1.3 | medium |
| `DelaySimulation` | Signal delay simulation | 0–50 samples | dangerous |
| `RandomTimeShift` | Random time shift | 0–±25 samples | dangerous |
| `SignalCrop` | Signal cropping from start | 2500–5000 samples | medium |

## Robustness categories

Categories based on experimental testing on LUDB dataset using RPN model (F1 baseline = 0.776):

- **safe** — F1 does not drop below 95% of baseline at any intensity
- **medium** — F1 drops below 90% only at higher intensity; use within recommended range
- **dangerous** — F1 drops below 90% at low intensity; use with restriction or avoid
- **critical** — causes complete model failure regardless of intensity

## License

MIT License
