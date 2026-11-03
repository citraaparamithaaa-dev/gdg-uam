"""Scale-independent time-series summaries, shared by training and inference.

No metadata, absolute channel level, or absolute channel scale is used.
"""
import numpy as np
from numba import njit


def fill_missing(series):
    raw = np.array(series, dtype=np.float64, copy=True)
    times = np.arange(raw.shape[1])
    for channel in raw:
        valid = np.isfinite(channel)
        if not valid.any():
            channel[:] = 0.0
        elif not valid.all():
            channel[~valid] = np.interp(times[~valid], times[valid], channel[valid])
    return raw


@njit(cache=True)
def time_features(raw):
    channels, length = raw.shape
    z = np.empty_like(raw)
    d = np.empty((channels, length - 1), dtype=np.float64)
    result = []
    for c in range(channels):
        center = raw[c].mean()
        scale = raw[c].std() + 1e-9
        z[c] = (raw[c] - center) / scale
        row = z[c]
        ordered = np.sort(row)
        for q in (0.0, .05, .25, .5, .75, .95, 1.0):
            idx = q * (length - 1)
            lo = int(idx)
            hi = min(lo + 1, length - 1)
            result.append(ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo))
        result.append(np.mean(row ** 3))
        result.append(np.mean(row ** 4))
        result.append(np.mean(np.abs(row)))
        result.append(np.mean((row[1:] * row[:-1]) < 0))
        delta = row[1:] - row[:-1]
        ds = delta.std() + 1e-9
        result.append(ds)
        result.append(np.mean(np.abs(delta)))
        result.append(np.max(np.abs(delta)))
        result.append(np.mean(((delta - delta.mean()) / ds) ** 3))
        result.append(np.mean(((delta - delta.mean()) / ds) ** 4))
        d[c] = (delta - delta.mean()) / ds
        for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            result.append(np.mean(row[lag:] * row[:-lag]))
        for block in range(4):
            segment = row[block * 64:(block + 1) * 64]
            result.append(segment.mean())
            result.append(segment.std())
    # Directional inter-channel lags and correlations of first differences.
    # First differences make short synchronous events visible despite slow drift.
    for variant in range(2):
        signals = z if variant == 0 else d
        n = signals.shape[1]
        for a in range(channels):
            for b in range(a + 1, channels):
                largest = -1e30
                smallest = 1e30
                best_lag = 0
                for lag in range(-16, 17):
                    start = max(0, -lag)
                    stop = min(n, n - lag)
                    corr = 0.0
                    for t in range(start, stop):
                        corr += signals[a, t] * signals[b, t + lag]
                    corr /= stop - start
                    result.append(corr)
                    if corr > largest:
                        largest = corr
                        best_lag = lag
                    smallest = min(smallest, corr)
                result.append(largest)
                result.append(smallest)
                result.append(float(best_lag))
    for a in range(channels):
        for b in range(a + 1, channels):
            for block in range(4):
                start = 64 * block
                result.append(np.mean(z[a, start:start+64] * z[b, start:start+64]))
    return np.asarray(result), z


def features(series):
    temporal, z = time_features(fill_missing(series))
    power = np.abs(np.fft.rfft(z, axis=1)) ** 2
    power[:, 0] = 0
    power /= power.sum(axis=1, keepdims=True) + 1e-12
    bins = np.arange(power.shape[1], dtype=float)
    edges = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 129)
    bands = np.stack([power[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    spectral = np.column_stack([
        np.log(bands + 1e-9),
        power.argmax(axis=1) / 256,
        power.max(axis=1),
        (power * bins).sum(axis=1) / 256,
        -(power * np.log(power + 1e-12)).sum(axis=1),
        np.sort(power, axis=1)[:, -3:].sum(axis=1),
    ])
    return np.nan_to_num(np.concatenate([temporal, spectral.ravel()]),
                         nan=0, posinf=0, neginf=0).astype(np.float64)


def basic_shape_indices():
    # First 144 entries: the 36 per-channel shape summaries above.
    # Raw zero-lag correlations, then spectral features, form the compact variant.
    ids = list(range(144))
    for pair in range(6):
        ids.append(144 + pair * 36 + 16)
    ids.extend(range(600, 672))
    return np.asarray(ids, dtype=np.int64)
