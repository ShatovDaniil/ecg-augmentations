# ecg-augmentations

Python library for ECG signal augmentation compatible with PyTorch.
Developed as part of a bachelor's thesis at VUT-FEKT (BUT-FEEC) (2026).

## Installation

```bash
pip install git+https://github.com/ShatovDaniil/ecg-augmentations.git
```

Or clone the full repository (required for the demo):

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
- h5py >= 3.0.0
- matplotlib >= 3.3.0

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

## Demo

To run the demo, clone the repository (the demo script and sample
signal file are not included in the pip package):

```bash
git clone https://github.com/ShatovDaniil/ecg-augmentations.git
cd ecg-augmentations
pip install -e .
python demo.py
```

The demo applies selected augmentations to a real clinical ECG recording
(`0000000b_00.h5`, SignalPlant 1.2.8.2, Fs = 2000 Hz) and plots
a comparison of the original and augmented signal for each augmentation.

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

### Summary of Robustness Thresholds and Recommended Augmentation Ranges

| Augmentation             | Recommended Range   | F1 at Threshold | Category         |
| :----------------------- |:--------------------| :-------------: | :--------------- |
| **ZeroOutSignal**        | 0–40 %              |      0.776      | Stable           |
| **ZeroOutChannel**       | 1 channel out of N  |      0.773      | Stable           |
| **ClippingDistortion**   | Unlimited           |      0.769      | Stable           |
| **HighPassFilter**       | 0.1–10 Hz           |      0.759      | Stable           |
| **DelaySimulation**      | Unlimited           |      0.776      | Stable           |
| **RandomTimeShift**      | Unlimited           |      0.776      | Stable           |
| **Scaling**              | Deviation ±1.0      |      0.739      | Moderately Stable|
| **RandomAmplitudeVariation**| Deviation ±1.0  |     0.735      | Moderately Stable|
| **SignalDrift (Linear)** | Amplitude 0–0.05    |      0.737      | Moderately Stable|
| **SignalDrift (Sinusoidal)**| Amplitude 0–0.05    |      0.734      | Moderately Stable|
| **BaselineWander**       | 0–0.05 mV           |      0.734      | Moderately Stable|
| **SinusoidalStretching** | Amplitude 0–0.5     |      0.707      | Moderately Stable|
| **EMGNoise**             | Amplitude 0–0.5     |      0.705      | Moderately Stable|
| **RandomNoise**          | 0–10 %              |      0.693      | Moderately Stable|
| **SignalArtefactAdder**  | With caution        |      0.682      | Moderately Stable|
| **RandomSignalReplacement**| 0–100 samples       |      0.729      | Moderately Stable|
| **SignalTimeStretch**    | Factor 0.9–1.3      |      0.695      | Moderately Stable|
| **SignalCrop**           | 2500–5000 samples   |      0.725      | Moderately Stable|
| **SignalInverter**       | Not recommended     |      0.661      | Unstable         |
| **PowerlineNoise**       | 0–0.02 mV           |      0.620      | Unstable         |
| **SignalDrift (Random)** | 0–0.002             |      0.722      | Unstable         |
| **EchoAdder**            | Not recommended     |      0.666      | Unstable         |
| **LowPassFilter**        | Not recommended     |      0.460      | Unstable         |
| **SegmentShuffle**       | Not recommended     |      0.405      | Unstable         |

---

### Augmentation Categorization by Stability

Augmentations are divided into three categories based on the degree of F1-score degradation compared to the baseline value:

* **Stable**
  The F1-score does not drop below 95% of the baseline value even at high intensity. The augmentation can be used without restrictions.
* **Moderately Stable**
  The F1-score drops below 90% of the baseline value only at higher intensities. The augmentation can be used within the recommended range specified in the table.
* **Unstable**
  The F1-score drops below 90% of the baseline value even at low intensity or causes structural damage to the signal. Usage is not recommended, or should be strictly limited.or causes structural damage to the signal. Usage is not recommended, or should be strictly limited.

## License

MIT License