"""
demo.py — demonstration of ecg-augmentations library.

Uses a synthetic ECG-like signal (sum of sinusoids) to visualize
the effect of each augmentation. No real data required.

Usage:
    python demo.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from augmentations import (
    RandomNoiseAdder, PowerlineNoise, BaselineWander, EMGNoise,
    HighPassFilter, LowPassFilter, Scaling, RandomAmplitudeVariation,
    ClippingDistortion, SignalInverter, SignalDrift, SinusoidalStretching,
    EchoAdder, SignalArtefactAdder, ZeroOutSignal, ZeroOutChannel,
    SegmentShuffle, SignalTimeStretch, DelaySimulation, RandomTimeShift,
    SignalCrop,
)

SAMPLING_F = 500
SIGNAL_LENGTH = 7700


def make_synthetic_ecg(n=SIGNAL_LENGTH, fs=SAMPLING_F):
    """
    Synthetic ECG-like signal composed of sinusoids.
    Approximates P wave, QRS complex and T wave periodicity.
    """
    t = np.linspace(0, n / fs, n)

    # Base components
    p_wave = 0.15 * np.sin(2 * np.pi * 1.2 * t)
    qrs = 0.80 * np.sin(2 * np.pi * 10 * t) * np.exp(-((t % (1/1.2) - 0.12)**2) / 0.0005)
    t_wave = 0.25 * np.sin(2 * np.pi * 1.2 * t + 1.8)
    baseline = 0.02 * np.sin(2 * np.pi * 0.15 * t)
    noise = np.random.randn(n) * 0.01

    signal = p_wave + qrs + t_wave + baseline + noise
    return torch.tensor(signal, dtype=torch.float32)


def plot_augmentation(x_orig, x_aug, title, filename=None):
    """Plots original and augmented signal side by side."""
    t_orig = np.arange(len(x_orig)) / SAMPLING_F
    t_aug = np.arange(len(x_aug)) / SAMPLING_F

    fig, axes = plt.subplots(2, 1, figsize=(12, 5))

    axes[0].plot(t_orig, x_orig.numpy(), color='steelblue', linewidth=0.8)
    axes[0].set_title('Původní signál', fontsize=11)
    axes[0].set_xlabel('t [s]')
    axes[0].set_ylabel('U [-]')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_aug, x_aug.numpy(), color='tomato', linewidth=0.8)
    axes[1].set_title(f'Po augmentaci: {title}', fontsize=11)
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('U [-]')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'  Saved: {filename}')
    else:
        plt.show()
    plt.close()


def run_demo(save_plots=False):
    """Runs demonstration of all augmentations."""

    torch.manual_seed(42)
    np.random.seed(42)

    x = make_synthetic_ecg()
    y = [(400, 420), (900, 920), (1400, 1420),
         (1900, 1920), (2400, 2420), (2900, 2920),
         (3400, 3420), (3900, 3920), (4400, 4420)]

    demos = [
        ("RandomNoise (10%)",
         RandomNoiseAdder(low_limit=0.0, high_limit=0.1, p=1.0, use_on='sample')),

        ("PowerlineNoise (0.02 mV)",
         PowerlineNoise(amplitude=0.02, sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("BaselineWander (0.3 mV)",
         BaselineWander(amplitude=0.3, sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("EMGNoise (0.3)",
         EMGNoise(amplitude=0.3, sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("HighPassFilter (1 Hz)",
         HighPassFilter(critical_f=1.0, sampling_f=SAMPLING_F, order=101, p=1.0, use_on='sample')),

        ("LowPassFilter (40 Hz)",
         LowPassFilter(critical_f=40.0, sampling_f=SAMPLING_F, order=101, p=1.0, use_on='sample')),

        ("Scaling (x0.5 -- x1.5)",
         Scaling(min_scale=0.5, max_scale=1.5, p=1.0, use_on='sample')),

        ("RandomAmplitudeVariation",
         RandomAmplitudeVariation(min_factor=0.7, max_factor=1.3,
                                  segment_count=5, p=1.0, use_on='sample')),

        ("ClippingDistortion (+-0.3 mV)",
         ClippingDistortion(clip_min=-0.3, clip_max=0.3, p=1.0, use_on='sample')),

        ("SignalInverter",
         SignalInverter(p=1.0, use_on='sample')),

        ("SignalDrift linear (0.2)",
         SignalDrift(drift_type='linear', amplitude=0.2, period=1.0,
                     sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("SignalDrift sinusoidal (0.2)",
         SignalDrift(drift_type='sinusoidal', amplitude=0.2, period=5.0,
                     sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("SignalDrift random (0.005)",
         SignalDrift(drift_type='random', amplitude=0.005, period=1.0,
                     sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("SinusoidalStretching (0.3)",
         SinusoidalStretching(amplitude=0.3, phase=0, period=5,
                              sampling_f=SAMPLING_F, p=1.0, use_on='sample')),

        ("EchoAdder (delay=50)",
         EchoAdder(delay=50, decay=0.3, p=1.0, use_on='sample')),

        ("SignalArtefactAdder",
         SignalArtefactAdder(p=1.0, use_on='sample')),

        ("ZeroOutSignal (10%)",
         ZeroOutSignal(max_zero_duration=0.1, p=1.0, use_on='sample')),

        ("ZeroOutChannel",
         ZeroOutChannel(p=1.0, use_on='sample')),

        ("SegmentShuffle (500 samples)",
         SegmentShuffle(segment_length=500, p=1.0, use_on='sample')),

        ("SignalTimeStretch (1.1)",
        SignalTimeStretch(stretch_factor=1.1, p=1.0, use_on='all')),

        ("DelaySimulation (max=100)",
         DelaySimulation(max_delay=100, p=1.0, use_on='all')),

        ("RandomTimeShift (+-50)",
         RandomTimeShift(shift_min=-50, shift_max=50, p=1.0, use_on='all')),

        ("SignalCrop (3000 samples)",
         SignalCrop(start=0, end=3000, p=1.0, use_on='all')),
    ]

    print("=" * 60)
    print("ecg-augmentations DEMO")
    print(f"Signal length: {SIGNAL_LENGTH} samples, fs={SAMPLING_F} Hz")
    print("=" * 60)

    for title, transform in demos:
        x_test = x.clone()
        y_test = list(y)
        x_aug, _ = transform(x_test, y_test, global_apply_to_all=True)

        filename = f"demo_{title.split('(')[0].strip().replace(' ', '_')}.png" if save_plots else None
        plot_augmentation(x, x_aug, title, filename)
        print(f"  {title:<40} out_len={len(x_aug)}")

    print("\nDemo completed.")


if __name__ == '__main__':
    # Set save_plots=True to save PNG files instead of showing interactively
    run_demo(save_plots=False)