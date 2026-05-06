import torch
import random
import numpy as np
from scipy import signal
import math


class BaseTransform:
    """Base class for all augmentation transforms."""

    def __init__(self, use_on='sample', apply_to_all_channels=False, p=0.1, loader=None):
        assert use_on in {'sample', 'target', 'all'}
        self.use_on = use_on
        self.apply_to_all_channels = apply_to_all_channels
        self.p = p
        self.loader = loader

    def __call__(self, x, y, global_apply_to_all):
        if global_apply_to_all or random.random() < self.p:
            return self.apply_transform(x, y)
        return x, y

    def apply_transform(self, x, y):
        if self.use_on == 'sample':
            return self.transform_x(x), y
        elif self.use_on == 'target':
            return x, self.transform_y(y)
        else:
            return self.transform_x(x), self.transform_y(y)

    def transform_x(self, x):
        return x

    def transform_y(self, y):
        return y


class SignalCrop(BaseTransform):
    """Crops the signal to a specified range [start, end]."""

    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
        self.start_idx = start
        self.end_idx = end

    def apply_transform(self, x, y):
        start = max(0, self.start_idx)
        end = min(len(x), self.end_idx)
        x_new = x[start:end]

        y_new = []
        for (m_start, m_end) in y:
            new_m_start = max(0, m_start - start)
            new_m_end = min(end - start, m_end - start)
            if new_m_start < new_m_end:
                y_new.append((new_m_start, new_m_end))

        if self.use_on == 'sample':
            return x_new, y
        elif self.use_on == 'target':
            return x, y_new
        else:
            return x_new, y_new


class RandomNoiseAdder(BaseTransform):
    """Adds Gaussian noise to the signal."""

    def __init__(self, low_limit: float, high_limit: float, **kwargs):
        super().__init__(**kwargs)
        self.low_limit = low_limit
        self.high_limit = high_limit

    def transform_x(self, x):
        power = self.high_limit
        if power == 0:
            return x
        noise = torch.randn_like(x) * power
        return x + noise


class PowerlineNoise(BaseTransform):
    """Adds 50 Hz powerline interference."""

    def __init__(self, amplitude: float, sampling_f: float, **kwargs):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.sampling_f = sampling_f

    def transform_x(self, x):
        t = torch.arange(len(x)) / self.sampling_f
        noise = self.amplitude * torch.sin(2 * math.pi * 50 * t)
        return x + noise


class BaselineWander(BaseTransform):
    """Adds low-frequency baseline drift simulating breathing (0.2 Hz)."""

    def __init__(self, amplitude: float, sampling_f: float, **kwargs):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.sampling_f = sampling_f

    def transform_x(self, x):
        t = torch.arange(len(x)) / self.sampling_f
        wander = self.amplitude * torch.sin(2 * math.pi * 0.2 * t)
        return x + wander


class EMGNoise(BaseTransform):
    """Adds high-frequency muscle noise (above 100 Hz)."""

    def __init__(self, amplitude: float, sampling_f: float, order: int = 101, **kwargs):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        fir_filter = signal.firwin(
            order, cutoff=100.0, window="nuttall", pass_zero=False, fs=sampling_f
        )
        self.filter = torch.tensor(fir_filter, dtype=torch.float32)

    def transform_x(self, x):
        noise = torch.randn_like(x) * self.amplitude
        filtered_noise = torch.nn.functional.conv1d(
            noise.unsqueeze(0).unsqueeze(0),
            self.filter.view(1, 1, -1),
            padding='same'
        ).squeeze()
        return x + filtered_noise


class HighPassFilter(BaseTransform):
    """FIR high-pass filter with Nuttall window."""

    def __init__(self, critical_f: float, sampling_f: int, order: int, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        fir_filter = signal.firwin(
            order, cutoff=critical_f, window="nuttall", pass_zero=False, fs=sampling_f
        )
        self.filter = torch.tensor(fir_filter, dtype=torch.float32)

    def transform_x(self, x):
        filtered = torch.nn.functional.conv1d(
            x.unsqueeze(0).unsqueeze(0),
            self.filter.view(1, 1, -1),
            padding='same'
        )
        return filtered.squeeze()


class LowPassFilter(BaseTransform):
    """FIR low-pass filter with Nuttall window."""

    def __init__(self, critical_f: float, sampling_f: int, order: int, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        fir_filter = signal.firwin(
            order, cutoff=critical_f, window="nuttall", pass_zero=True, fs=sampling_f
        )
        self.filter = torch.tensor(fir_filter, dtype=torch.float32)

    def transform_x(self, x):
        pad = self.order // 2
        x_padded = torch.cat([x[:pad].flip(0), x, x[-pad:].flip(0)])
        filtered = torch.nn.functional.conv1d(
            x_padded.unsqueeze(0).unsqueeze(0),
            self.filter.view(1, 1, -1),
            padding=0
        )
        return filtered.squeeze()[pad:-pad]


class Scaling(BaseTransform):
    """Scales the signal by a random factor in [min_scale, max_scale]."""

    def __init__(self, min_scale: float, max_scale: float, **kwargs):
        super().__init__(**kwargs)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def transform_x(self, x):
        scale_factor = torch.rand(1).item() * (self.max_scale - self.min_scale) + self.min_scale
        return x * scale_factor


class RandomAmplitudeVariation(BaseTransform):
    """Applies random amplitude scaling independently to each segment."""

    def __init__(self, min_factor: float, max_factor: float, segment_count: int, **kwargs):
        super().__init__(**kwargs)
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.segment_count = segment_count

    def transform_x(self, x):
        segment_length = len(x) // self.segment_count
        augmented = x.clone()
        for i in range(self.segment_count):
            start = i * segment_length
            end = start + segment_length if i < self.segment_count - 1 else len(x)
            factor = torch.rand(1).item() * (self.max_factor - self.min_factor) + self.min_factor
            augmented[start:end] *= factor
        return augmented


class ClippingDistortion(BaseTransform):
    """Clips signal amplitude to [clip_min, clip_max], simulating amplifier saturation."""

    def __init__(self, clip_min: float, clip_max: float, **kwargs):
        super().__init__(**kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def transform_x(self, x):
        return torch.clamp(x, min=self.clip_min, max=self.clip_max)


class SignalInverter(BaseTransform):
    """Inverts the polarity of the signal (multiplies by -1)."""

    def transform_x(self, x):
        return -x


class SignalDrift(BaseTransform):
    """Adds baseline drift: linear, sinusoidal, exponential, or random."""

    def __init__(self, drift_type: str, amplitude: float, period: float, sampling_f: float, **kwargs):
        super().__init__(**kwargs)
        self.drift_type = drift_type
        self.amplitude = amplitude
        self.period = period
        self.sampling_f = sampling_f

    def transform_x(self, x):
        t = torch.arange(len(x)) / self.sampling_f
        if self.drift_type == 'linear':
            drift = self.amplitude * t / t.max()
        elif self.drift_type == 'exponential':
            drift = self.amplitude * (torch.exp(t / t.max()) - 1)
        elif self.drift_type == 'sinusoidal':
            drift = self.amplitude * torch.sin(2 * math.pi * (1 / self.period) * t)
        elif self.drift_type == 'random':
            drift = (torch.randn_like(t) * self.amplitude).cumsum(dim=0)
        else:
            drift = torch.zeros_like(t)
        return x + drift


class SinusoidalStretching(BaseTransform):
    """Modulates signal amplitude with a sinusoidal pattern."""

    def __init__(self, amplitude: float, phase: float, period: float, sampling_f: float, **kwargs):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.phase = phase
        self.period = period
        self.sampling_f = sampling_f

    def transform_x(self, x):
        t = torch.arange(len(x)) / self.sampling_f
        pattern = 1 + self.amplitude * torch.sin(2 * torch.pi * (1 / self.period) * t + self.phase)
        return x * pattern


class EchoAdder(BaseTransform):
    """Adds a delayed and attenuated copy of the signal (echo effect)."""

    def __init__(self, delay: float, decay: float, **kwargs):
        super().__init__(**kwargs)
        self.delay = int(delay)
        self.decay = decay

    def transform_x(self, x):
        echo = torch.zeros_like(x)
        if self.delay > 0 and self.delay < len(x):
            echo[self.delay:] = x[:-self.delay] * self.decay
        return x + echo


class SignalArtefactAdder(BaseTransform):
    """Adds random impulse artefacts simulating electrode movement."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rng = torch.Generator()

    def transform_x(self, x):
        num_impulses = torch.randint(1, 50, (1,), generator=self.rng).item()
        augmented = x.clone()
        for _ in range(num_impulses):
            pos = torch.randint(0, len(x), (1,), generator=self.rng).item()
            amp = torch.rand(1, generator=self.rng).item() * 2
            augmented[pos] += amp
        return augmented


class ZeroOutSignal(BaseTransform):
    """Zeros out random segments of the signal, simulating dropouts."""

    def __init__(self, max_zero_duration: float, **kwargs):
        super().__init__(**kwargs)
        self.max_zero_duration = max_zero_duration
        self.rng = torch.Generator()

    def transform_x(self, x):
        num_zeros = torch.randint(1, 4, (1,), generator=self.rng).item()
        length = len(x)
        max_dur = min(self.max_zero_duration, 1.0)
        for _ in range(num_zeros):
            dur = int(max_dur * length * torch.rand(1, generator=self.rng).item())
            dur = max(1, dur)
            start = torch.randint(0, max(length - dur, 1), (1,), generator=self.rng).item()
            x[start:start + dur] = 0
        return x


class ZeroOutChannel(BaseTransform):
    """Zeros out the entire signal channel."""

    def transform_x(self, x):
        return x * 0


class SegmentShuffle(BaseTransform):
    """Randomly reorders fixed-length segments of the signal."""

    def __init__(self, segment_length: int, **kwargs):
        super().__init__(**kwargs)
        self.segment_length = segment_length

    def transform_x(self, x):
        num_segments = len(x) // self.segment_length
        if num_segments == 0:
            return x
        segments = [x[i * self.segment_length:(i + 1) * self.segment_length] for i in range(num_segments)]
        segments = torch.stack(segments, dim=0)
        shuffled = segments[torch.randperm(segments.size(0))]
        tail = x[num_segments * self.segment_length:]
        return torch.cat([shuffled.flatten(), tail])


class RandomSignalReplacement(BaseTransform):
    """Replaces a random segment with a segment from another LUDB record."""

    def __init__(self, data_dir: str, min_segment_length: int = 100,
                 max_segment_length: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        import os
        self.files = [
            os.path.join(data_dir, f[:-4])
            for f in os.listdir(data_dir) if f.endswith('.dat')
        ]

    def transform_x(self, x):
        if len(self.files) == 0:
            return x
        import wfdb
        try:
            donor_file = random.choice(self.files)
            signals, _ = wfdb.rdsamp(donor_file, channels=[0])
            donor = torch.tensor(signals[:, 0], dtype=torch.float32)
            length = random.randint(self.min_segment_length, self.max_segment_length)
            max_start = max(0, min(len(x), len(donor)) - length)
            if max_start == 0:
                return x
            start = random.randint(0, max_start)
            end = min(start + length, len(x), len(donor))
            x = x.clone()
            x[start:end] = donor[start:end]
        except Exception:
            pass
        return x


class SignalTimeStretch(BaseTransform):
    """Stretches or compresses the signal in time via linear interpolation."""

    def __init__(self, stretch_factor: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.stretch_factor = stretch_factor

    def apply_transform(self, x, y):
        n = len(x)
        new_length = int(n * self.stretch_factor)
        new_time = np.linspace(0, n - 1, num=new_length)
        stretched = np.interp(new_time, np.arange(n), x.numpy())
        x_new = torch.tensor(stretched, dtype=torch.float32)

        y_new = [(int(s * self.stretch_factor), int(e * self.stretch_factor)) for s, e in y]

        if self.use_on == 'sample':
            return x_new, y
        elif self.use_on == 'target':
            return x, y_new
        else:
            return x_new, y_new


class DelaySimulation(BaseTransform):
    """Simulates signal delay by shifting with zero-padding at the start."""

    def __init__(self, max_delay: int, **kwargs):
        super().__init__(**kwargs)
        self.max_delay = int(max_delay)

    def apply_transform(self, x, y):
        delay = torch.randint(0, self.max_delay + 1, (1,)).item()
        if delay == 0:
            return x, y

        delayed = torch.zeros_like(x)
        delayed[delay:] = x[:-delay]

        y_new = []
        for (s, e) in y:
            ns = min(s + delay, len(x))
            ne = min(e + delay, len(x))
            if ns < ne:
                y_new.append((ns, ne))

        if self.use_on == 'sample':
            return delayed, y
        elif self.use_on == 'target':
            return x, y_new
        else:
            return delayed, y_new


class RandomTimeShift(BaseTransform):
    """Shifts the signal in time by a random value in [shift_min, shift_max]."""

    def __init__(self, shift_min: int, shift_max: int, **kwargs):
        super().__init__(**kwargs)
        self.shift_min = shift_min
        self.shift_max = shift_max

    def apply_transform(self, x, y):
        shift = torch.randint(self.shift_min, self.shift_max + 1, (1,)).item()
        if shift == 0:
            return x, y

        shifted = torch.zeros_like(x)
        if shift > 0:
            shifted[shift:] = x[:-shift]
        else:
            shifted[:shift] = x[-shift:]

        y_new = []
        for (s, e) in y:
            ns = max(0, min(len(x), s + shift))
            ne = max(0, min(len(x), e + shift))
            if ns < ne:
                y_new.append((ns, ne))

        if self.use_on == 'sample':
            return shifted, y
        elif self.use_on == 'target':
            return x, y_new
        else:
            return shifted, y_new