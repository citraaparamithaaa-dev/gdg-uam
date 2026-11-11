"""Vectorized signal features for CPU inference without runtime compilation.

Only the slots used by the selected classifier are computed. Every call owns
its working arrays, so read-only inputs and repeated calls are safe.
"""
import numpy as np

TIME = np.arange(256, dtype=np.float64) - 127.5
TIME_ENERGY = TIME @ TIME
POSITIONS = np.arange(256)
LAGS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
AMP_LAGS = (1, 4, 16, 32, 64)
EDGES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 129)
BINS = np.arange(129)
QPOS = np.array([0., .05, .25, .5, .75, .95, 1.]) * 255
QLOW = QPOS.astype(np.int64)
QHIGH = np.minimum(QLOW + 1, 255)
QFRAC = QPOS - QLOW
STARTS = (0, 64, 128, 0, 64, 128, 192)
WIDTHS = (128, 128, 128, 64, 64, 64, 64)
WINDOWS = {width: np.hanning(width) for width in (64, 128)}
RETAINED = [0, 1, 4]


def preprocess(series):
    """Interpolate spikes and missing values in a private float64 copy."""
    raw = np.array(series, dtype=np.float64, copy=True)
    for row in raw:
        valid = np.isfinite(row)
        count = np.count_nonzero(valid)
        if count == 0:
            row[:] = 0.
            continue
        if count >= 20:
            values = row[valid]
            median = np.median(values)
            mad = np.median(np.abs(values - median)) * 1.4826 + 1e-9
            bad = valid & (np.abs((row - median) / mad) > 4.5)
            good = valid & ~bad
            if np.count_nonzero(good) >= 10 and np.any(bad):
                row[bad] = np.interp(POSITIONS[bad], POSITIONS[good], row[good])
        if count < 256:
            row[~valid] = np.interp(POSITIONS[~valid], POSITIONS[valid], row[valid])
    return raw


def quantile_sorted(ordered, q):
    """Match NumPy's linear quantile arithmetic using an already sorted row."""
    position = q * (ordered.shape[1] - 1)
    lower = int(position)
    upper = min(lower + 1, ordered.shape[1] - 1)
    fraction = position - lower
    a, b = ordered[:, lower], ordered[:, upper]
    if fraction < .5:
        return a + (b - a) * fraction
    return b - (b - a) * (1 - fraction)


def features(series):
    raw = preprocess(series)
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    out = np.zeros(417, dtype=np.float64)
    first = z[:2]
    ordered = np.sort(first, axis=1)
    delta = np.diff(first, axis=1)
    ds = delta.std(axis=1) + 1e-9
    normalized_delta = (delta - delta.mean(axis=1, keepdims=True)) / ds[:, None]
    base = out[:72].reshape(2, 36)
    base[:, :7] = ordered[:, QLOW] * (1 - QFRAC) + ordered[:, QHIGH] * QFRAC
    base[:, 7:16] = np.column_stack((
        (first ** 3).mean(axis=1), (first ** 4).mean(axis=1),
        np.abs(first).mean(axis=1), (first[:, 1:] * first[:, :-1] < 0).mean(axis=1),
        ds, np.abs(delta).mean(axis=1), np.abs(delta).max(axis=1),
        (normalized_delta ** 3).mean(axis=1), (normalized_delta ** 4).mean(axis=1)))
    for i, lag in enumerate(LAGS):
        base[:, 16 + i] = (first[:, lag:] * first[:, :-lag]).mean(axis=1)
    blocks = first.reshape(2, 4, 64)
    base[:, 28::2] = blocks.mean(axis=2)
    base[:, 29::2] = blocks.std(axis=2)
    out[144:150] = [(z[a] * z[b]).mean() for a in range(4) for b in range(a + 1, 4)]

    power = np.abs(np.fft.rfft(first, axis=1)) ** 2
    power[:, 0] = 0
    power /= power.sum(axis=1, keepdims=True) + 1e-12
    spectral = out[150:186].reshape(2, 18)
    for i, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        spectral[:, i] = np.log(power[:, a:b].sum(axis=1) + 1e-9)
    spectral[:, 13:] = np.column_stack((
        power.argmax(axis=1) / 256, power.max(axis=1),
        (power * BINS).sum(axis=1) / 256,
        -(power * np.log(power + 1e-12)).sum(axis=1),
        np.sort(power, axis=1)[:, -3:].sum(axis=1)))

    signals = np.concatenate((z, (z[:1] + z[1:2]) / 2), axis=0)
    median = np.median(signals, axis=1, keepdims=True)
    mad = np.median(np.abs(signals - median), axis=1, keepdims=True) * 1.4826 + 1e-6
    signals = np.clip(signals, median - 4 * mad, median + 4 * mad)
    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True) + 1e-9
    spectrum = np.fft.fft(signals[[0, 1, 3, 4]], axis=1)
    spectrum[:, 0] = 0
    spectrum[:, 128:] = 0
    spectrum[:, 1:128] *= 2
    analytic = np.fft.ifft(spectrum, axis=1)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic[[0, 1, 3]]), axis=1)
    phase -= phase.mean(axis=1, keepdims=True)
    pad = np.zeros((5, 256))
    pad[RETAINED] = phase
    slopes = (pad @ TIME / TIME_ENERGY)[RETAINED]
    residual = phase - slopes[:, None] * TIME
    instant = np.diff(phase, axis=1)
    amp = amplitude / (amplitude.mean(axis=1, keepdims=True) + 1e-9)
    amp3 = amp[[0, 1, 3]]
    pad[RETAINED] = amp3
    amp_slopes = (pad @ TIME / TIME_ENERGY)[RETAINED]
    ordered_instant = np.sort(instant, axis=1)
    ordered_amp = np.sort(amp3, axis=1)
    extra = np.zeros((3, 39))
    extra[:, :12] = np.column_stack((
        slopes, residual.std(axis=1), np.abs(residual).mean(axis=1),
        instant.std(axis=1), np.median(ordered_instant, axis=1),
        quantile_sorted(ordered_instant, .9) - quantile_sorted(ordered_instant, .1),
        instant[:, :127].mean(axis=1), instant[:, 127:].mean(axis=1),
        amp3.std(axis=1), (amp3 ** 3).mean(axis=1),
        quantile_sorted(ordered_amp, .9) - quantile_sorted(ordered_amp, .1), amp_slopes))
    centered = amp3 - 1
    energy = (centered ** 2).mean(axis=1) + 1e-9
    for i, lag in enumerate(AMP_LAGS):
        extra[:, 12 + i] = (centered[:, lag:] * centered[:, :-lag]).mean(axis=1) / energy
    channel3 = amp[2] - 1
    extra[:, 38] = (centered * channel3).mean(axis=1) / np.sqrt(energy * ((channel3 ** 2).mean() + 1e-9))

    # Transform the 21 retained windows together, avoiding seven separate FFTs.
    selected = signals[RETAINED]
    matrix = np.zeros((3, 7, 512))
    for j, (start, width) in enumerate(zip(STARTS, WIDTHS)):
        segment = selected[:, start:start + width]
        matrix[:, j, :width] = (segment - segment.mean(axis=1, keepdims=True)) * WINDOWS[width]
    power = np.abs(np.fft.rfft(matrix, axis=2)) ** 2
    power[:, :, 0] = 0
    power /= power.sum(axis=2, keepdims=True) + 1e-12
    extra[:, 17:38:3] = power.argmax(axis=2) / 512
    extra[:, 18:38:3] = power.max(axis=2)
    extra[:, 19:38:3] = -(power * np.log(power + 1e-12)).sum(axis=2)
    out[222:300] = extra[:2].ravel()
    out[378:417] = extra[2]
    return np.nan_to_num(out, nan=0., posinf=0., neginf=0.)
