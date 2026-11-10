"""Pruned extractor for the balanced_core_200 feature set.

Computes only the 231 features the exported classifier reads, in the same
417-slot layout as structure_features.features, with unused slots zeroed.
Every retained value is produced by arithmetic identical to the original
implementation (same operations in the same order), so predictions are
unchanged. Reductions use a bitwise replica of NumPy's pairwise summation,
and the two matrix-vector products keep the original (5, 256) shape with
unused rows zeroed so BLAS accumulation order is preserved.

Per-signal summary layout (39 slots, from structure_features.extra_features):
 0 slope, 1 residual.std, 2 mean|residual|, 3 std(instant), 4 median(instant),
 5 q90-q10(instant), 6 mean instant[:127], 7 mean instant[127:], 8 amp.std,
 9 mean(amp^3), 10 q90-q10(amp), 11 amp slope, 12-16 amp autocorr lags
 (1,4,16,32,64), 17-37 seven segment spectra (peak bin, peak power, entropy),
 38 amplitude correlation with channel 3.
"""
import numpy as np
from numba import njit

TIME = np.arange(256, dtype=float) - 127.5
TIME_ENERGY = np.dot(TIME, TIME)
SPECTRAL_EDGES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 129)
SEGMENT_STARTS = np.array([0, 64, 128, 0, 64, 128, 192], dtype=np.int64)
SEGMENT_WIDTHS = np.array([128, 128, 128, 64, 64, 64, 64], dtype=np.int64)
WINDOW_128 = np.hanning(128)
WINDOW_64 = np.hanning(64)


@njit(cache=True)
def _pairwise_sum(a):
    n = a.shape[0]
    if n < 8:
        r = 0.0
        for i in range(n):
            r += a[i]
        return r
    if n <= 128:
        r0 = a[0]; r1 = a[1]; r2 = a[2]; r3 = a[3]
        r4 = a[4]; r5 = a[5]; r6 = a[6]; r7 = a[7]
        i = 8
        lim = n - (n % 8)
        while i < lim:
            r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise_sum(a[:n2]) + _pairwise_sum(a[n2:])


@njit(cache=True)
def _mean(a):
    return _pairwise_sum(a) / a.shape[0]


@njit(cache=True)
def _std(a):
    m = _mean(a)
    n = a.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = (a[i] - m) ** 2
    return np.sqrt(_pairwise_sum(d) / n)


@njit(cache=True)
def _quantile_sorted(s, q):
    n = s.shape[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    t = idx - lo
    av = s[lo]
    bv = s[hi]
    if t < 0.5:
        return av + (bv - av) * t
    return bv - (bv - av) * (1 - t)


@njit(cache=True)
def _median_sorted(s):
    n = s.shape[0]
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


@njit(cache=True)
def despike(series, k):
    """Replace samples beyond k robust deviations (median/MAD) with the channel
    median. The evaluation stream contains sparse class-correlated outliers;
    removing them restores the training-like signal shape. Numpy-equivalent:
    med = np.median(v); mad = np.median(|v-med|)*1.4826+1e-9; |z|>k -> med."""
    channels, length = series.shape
    out = np.empty((channels, length), dtype=np.float64)
    buf = np.empty(length, dtype=np.float64)
    dev = np.empty(length, dtype=np.float64)
    for c in range(channels):
        n = 0
        for t in range(length):
            x = series[c, t]
            out[c, t] = x
            if not np.isnan(x):
                buf[n] = x
                n += 1
        if n < 20:
            continue
        med = _median_sorted(np.sort(buf[:n]))
        for i in range(n):
            d = buf[i] - med
            dev[i] = -d if d < 0.0 else d
        mad = _median_sorted(np.sort(dev[:n])) * 1.4826 + 1e-9
        for t in range(length):
            x = out[c, t]
            if np.isnan(x):
                continue
            d = (x - med) / mad
            ad = -d if d < 0.0 else d
            if ad > k:
                out[c, t] = med
    return out


@njit(cache=True)
def despike_interp(series, k):
    """Replace samples beyond k robust deviations (median/MAD) with linear
    interpolation between non-outlier neighbors. Numpy-equivalent:
    med = np.median(v); mad = np.median(|v-med|)*1.4826+1e-9;
    outlier = |z| > k (finite only); out[outlier] = np.interp(...)."""
    channels, length = series.shape
    out = np.empty((channels, length), dtype=np.float64)
    buf = np.empty(length, dtype=np.float64)
    dev = np.empty(length, dtype=np.float64)
    mark = np.empty(length, dtype=np.int64)
    kp = np.empty(length, dtype=np.float64)
    kv = np.empty(length, dtype=np.float64)
    for c in range(channels):
        n = 0
        for t in range(length):
            x = series[c, t]
            out[c, t] = x
            if not np.isnan(x):
                buf[n] = x
                n += 1
        if n < 20:
            continue
        med = _median_sorted(np.sort(buf[:n]))
        for i in range(n):
            d = buf[i] - med
            dev[i] = -d if d < 0.0 else d
        mad = _median_sorted(np.sort(dev[:n])) * 1.4826 + 1e-9
        m = 0
        for t in range(length):
            x = out[c, t]
            if not np.isnan(x):
                d = (x - med) / mad
                ad = -d if d < 0.0 else d
                if ad > k:
                    mark[t] = 1
                    continue
                kp[m] = t
                kv[m] = x
                m += 1
            mark[t] = 0
        if m < 10:
            continue
        for t in range(length):
            if mark[t] == 1:
                out[c, t] = np.interp(float(t), kp[:m], kv[:m])
    return out


@njit(cache=True)
def fill_missing(series):
    channels, length = series.shape
    raw = np.empty((channels, length), dtype=np.float64)
    for c in range(channels):
        for t in range(length):
            raw[c, t] = series[c, t]
    for c in range(channels):
        channel = raw[c]
        n_valid = 0
        for t in range(length):
            if not np.isnan(channel[t]):
                n_valid += 1
        if n_valid == 0:
            for t in range(length):
                channel[t] = 0.0
        elif n_valid < length:
            xp = np.empty(n_valid)
            fp = np.empty(n_valid)
            pos = 0
            for t in range(length):
                if not np.isnan(channel[t]):
                    xp[pos] = t
                    fp[pos] = channel[t]
                    pos += 1
            for t in range(length):
                if np.isnan(channel[t]):
                    channel[t] = np.interp(float(t), xp, fp)
    return raw


@njit(cache=True)
def base_stats(raw, out):
    """Channels 0-1 shape summaries (out[0:72]) and the six zero-lag channel
    correlations (out[144:150]); statements mirror signal_features.time_features."""
    channels, length = raw.shape
    z = np.empty_like(raw)
    for c in range(channels):
        center = raw[c].mean()
        scale = raw[c].std() + 1e-9
        z[c] = (raw[c] - center) / scale
    pos = 0
    for c in range(2):
        row = z[c]
        ordered = np.sort(row)
        for q in (0.0, .05, .25, .5, .75, .95, 1.0):
            idx = q * (length - 1)
            lo = int(idx)
            hi = min(lo + 1, length - 1)
            out[pos] = ordered[lo] * (1 - (idx - lo)) + ordered[hi] * (idx - lo)
            pos += 1
        out[pos] = np.mean(row ** 3); pos += 1
        out[pos] = np.mean(row ** 4); pos += 1
        out[pos] = np.mean(np.abs(row)); pos += 1
        out[pos] = np.mean((row[1:] * row[:-1]) < 0); pos += 1
        delta = row[1:] - row[:-1]
        ds = delta.std() + 1e-9
        out[pos] = ds; pos += 1
        out[pos] = np.mean(np.abs(delta)); pos += 1
        out[pos] = np.max(np.abs(delta)); pos += 1
        out[pos] = np.mean(((delta - delta.mean()) / ds) ** 3); pos += 1
        out[pos] = np.mean(((delta - delta.mean()) / ds) ** 4); pos += 1
        for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            out[pos] = np.mean(row[lag:] * row[:-lag]); pos += 1
        for block in range(4):
            segment = row[block * 64:(block + 1) * 64]
            out[pos] = segment.mean(); pos += 1
            out[pos] = segment.std(); pos += 1
    p = 144
    for a in range(channels):
        for b in range(a + 1, channels):
            corr = 0.0
            for t in range(length):
                corr += z[a, t] * z[b, t]
            out[p] = corr / length
            p += 1
    return z


@njit(cache=True)
def clip_and_rescale(signals):
    """Median/MAD clip and per-row rescale, mirroring extra_features."""
    rows, length = signals.shape
    out = np.empty_like(signals)
    for r in range(rows):
        row = signals[r]
        med = _median_sorted(np.sort(row))
        dev = np.empty(length)
        for i in range(length):
            v = row[i] - med
            dev[i] = -v if v < 0.0 else v
        mad = _median_sorted(np.sort(dev)) * 1.4826 + 1e-6
        lo = med - 4 * mad
        hi = med + 4 * mad
        for i in range(length):
            v = row[i]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            out[r, i] = v
    for r in range(rows):
        row = out[r]
        m = _mean(row)
        for i in range(row.shape[0]):
            row[i] = row[i] - m
        # The original divides by the std of the already-centered row, so the
        # std here must also see the centered values.
        s = _std(row) + 1e-9
        for i in range(row.shape[0]):
            row[i] = row[i] / s
    return out


AMP_LAGS = np.array([1, 4, 16, 32, 64], dtype=np.int64)


@njit(cache=True)
def amp_lag_stats(amp3, amp_centered_s3, energy_s3, out):
    """Amp autocorrelation lags (out[:, 0:5]) and the channel-3 amplitude
    correlation (out[:, 5]) for the three retained signals."""
    rows, length = amp3.shape
    for r in range(rows):
        amp = amp3[r]
        centered = np.empty(length)
        for i in range(length):
            centered[i] = amp[i] - 1.0
        energy = _mean(centered * centered) + 1e-9
        for j in range(5):
            lag = AMP_LAGS[j]
            products = np.empty(length - lag)
            for i in range(length - lag):
                products[i] = centered[lag + i] * centered[i]
            out[r, j] = _mean(products) / energy
        cross = np.empty(length)
        for i in range(length):
            cross[i] = centered[i] * amp_centered_s3[i]
        out[r, 5] = _mean(cross) / np.sqrt(energy * energy_s3)
    return out


@njit(cache=True)
def summary_stats(phase3, slope, amp3, out):
    """Phase/amplitude summaries that need no matrix product, power or FFT.
    out columns follow extra_features summary order 1-8 and 10:
    1 residual.std, 2 mean|residual|, 3 std(instant), 4 median(instant),
    5 q90-q10(instant), 6 mean instant[:127], 7 mean instant[127:],
    8 amp.std, then (column 8) q90-q10(amp)."""
    rows, length = phase3.shape
    for r in range(rows):
        ph = phase3[r]
        residual = np.empty(length)
        for i in range(length):
            residual[i] = ph[i] - slope[r] * TIME[i]
        out[r, 0] = _std(residual)
        absr = np.empty(length)
        for i in range(length):
            v = residual[i]
            absr[i] = -v if v < 0.0 else v
        out[r, 1] = _mean(absr)
        instant = np.empty(length - 1)
        for i in range(length - 1):
            instant[i] = ph[i + 1] - ph[i]
        out[r, 2] = _std(instant)
        sorted_instant = np.sort(instant)
        out[r, 3] = _median_sorted(sorted_instant)
        out[r, 4] = (_quantile_sorted(sorted_instant, 0.9)
                     - _quantile_sorted(sorted_instant, 0.1))
        head = np.empty(127)
        for i in range(127):
            head[i] = instant[i]
        tail = np.empty(128)
        for i in range(128):
            tail[i] = instant[127 + i]
        out[r, 5] = _mean(head)
        out[r, 6] = _mean(tail)
        out[r, 7] = _std(amp3[r])
        sorted_amp = np.sort(amp3[r])
        out[r, 8] = (_quantile_sorted(sorted_amp, 0.9)
                     - _quantile_sorted(sorted_amp, 0.1))
    return out


@njit(cache=True)
def windowed_segments(signals3, matrix):
    """Zero-padded windowed segments for the three retained signals,
    7 segments each, into one (21, 512) matrix."""
    for r in range(3):
        row = signals3[r]
        for j in range(7):
            start = SEGMENT_STARTS[j]
            width = SEGMENT_WIDTHS[j]
            m = _mean(row[start:start + width])
            out = matrix[r * 7 + j]
            for i in range(width):
                win = WINDOW_128[i] if width == 128 else WINDOW_64[i]
                out[i] = (row[start + i] - m) * win
    return matrix


def spectral_stats(z2):
    """Spectral summaries for the first two channels, mirroring the tail of
    signal_features.features."""
    power = np.abs(np.fft.rfft(z2, axis=1)) ** 2
    power[:, 0] = 0
    power /= power.sum(axis=1, keepdims=True) + 1e-12
    bins = np.arange(power.shape[1], dtype=float)
    edges = SPECTRAL_EDGES
    bands = np.stack([power[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return np.column_stack([
        np.log(bands + 1e-9),
        power.argmax(axis=1) / 256,
        power.max(axis=1),
        (power * bins).sum(axis=1) / 256,
        -(power * np.log(power + 1e-12)).sum(axis=1),
        np.sort(power, axis=1)[:, -3:].sum(axis=1),
    ]).ravel()


def features(series):
    raw = fill_missing(series)
    base = np.zeros(222)
    z = base_stats(raw, base)
    base[150:186] = spectral_stats(z[[0, 1]])

    z2 = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    signals = np.concatenate([z2, (z2[0:1] + z2[1:2]) / 2], axis=0)
    signals = clip_and_rescale(signals)

    # Rows kept for the analytic stage: signals 0, 1, 3 and the shared 0/1 mean.
    kept = signals[[0, 1, 3, 4]]
    spectrum = np.fft.fft(kept, axis=1)
    spectrum[:, 0] = 0
    spectrum[:, 128:] = 0
    spectrum[:, 1:128] *= 2
    analytic = np.fft.ifft(spectrum, axis=1)
    amplitude = np.abs(analytic)
    phase3 = np.unwrap(np.angle(analytic[[0, 1, 3]]), axis=1)
    phase3 -= phase3.mean(axis=1, keepdims=True)

    # Matrix products keep the original (5, 256) shape with unused rows zeroed,
    # so BLAS accumulates each retained row in the same order.
    pad = np.zeros((5, 256))
    pad[[0, 1, 4]] = phase3
    slope = (pad @ TIME / TIME_ENERGY)[[0, 1, 4]]

    amp3 = amplitude[[0, 1, 3]]
    amp3 = amp3 / (amp3.mean(axis=1, keepdims=True) + 1e-9)
    amp_pad = np.zeros((5, 256))
    amp_pad[[0, 1, 4]] = amp3
    amp_slope = (amp_pad @ TIME / TIME_ENERGY)[[0, 1, 4]]

    amp_s3 = amplitude[2] / (amplitude[2].mean() + 1e-9)
    centered_s3 = amp_s3 - 1.0
    energy_s3 = float(np.mean(centered_s3 ** 2) + 1e-9)
    lags = np.empty((3, 6))
    amp_lag_stats(amp3, centered_s3, energy_s3, lags)
    core = np.empty((3, 9))
    summary_stats(phase3, slope, amp3, core)

    matrix = np.zeros((21, 512))
    windowed_segments(signals[[0, 1, 4]], matrix)
    power = np.abs(np.fft.rfft(matrix, axis=1)) ** 2
    power[:, 0] = 0
    power /= power.sum(axis=1, keepdims=True) + 1e-12
    seg = np.column_stack([power.argmax(axis=1) / 512,
                           power.max(axis=1),
                           -(power * np.log(power + 1e-12)).sum(axis=1)])

    extra = np.zeros((3, 39))
    extra[:, 0] = slope
    extra[:, 1:9] = core[:, :8]
    extra[:, 9] = np.mean(amp3 ** 3, axis=1)
    extra[:, 10] = core[:, 8]
    extra[:, 11] = amp_slope
    extra[:, 12:17] = lags[:, :5]
    extra[:, 17:38] = seg.reshape(3, 21)
    extra[:, 38] = lags[:, 5]

    result = np.zeros(195)
    result[0:39] = extra[0]
    result[39:78] = extra[1]
    result[156:195] = extra[2]
    out = np.zeros(417)
    out[:222] = np.nan_to_num(base, nan=0., posinf=0., neginf=0.)
    out[222:] = np.nan_to_num(result, nan=0., posinf=0., neginf=0.)
    return out
