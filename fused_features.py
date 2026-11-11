"""Fuse reductions into compiled stages, leaving FFTs in NumPy's optimized FFT.

Modes: 0 = basic shape/spectrum, 1 = add analytic modulation, 2 = full core.
The feature layout remains 417 slots so existing numerical trees can be reused.
"""
import numpy as np
from numba import njit
from core_features import (fill_missing, base_stats, clip_and_rescale, _mean, _std,
    _pairwise_sum, amp_lag_stats, summary_stats, windowed_segments, TIME, TIME_ENERGY)


@njit(cache=True)
def prepare(series, mode):
    raw = fill_missing(series)
    out = np.zeros(417)
    z = base_stats(raw, out)
    first = z[:2].copy()
    if mode == 0:
        return out, first, np.zeros((5, 256))
    signals = np.empty((5, 256))
    for c in range(4):
        signals[c] = (raw[c]-_mean(raw[c]))/(_std(raw[c])+1e-9)
    signals[4] = (signals[0]+signals[1])/2
    return out, first, clip_and_rescale(signals)


@njit(cache=True)
def finish_spectral(power, out):
    edges = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 129)
    for c in range(2):
        p = power[c]
        p[0] = 0
        p /= _pairwise_sum(p)+1e-12
        offset = 150+c*18
        for j in range(13):
            out[offset+j] = np.log(_pairwise_sum(p[edges[j]:edges[j+1]])+1e-9)
        out[offset+13] = np.argmax(p)/256
        out[offset+14] = np.max(p)
        out[offset+15] = _pairwise_sum(p*np.arange(129))/256
        out[offset+16] = -_pairwise_sum(p*np.log(p+1e-12))
        out[offset+17] = np.sort(p)[-3:].sum()


@njit(cache=True)
def phase_inputs(analytic):
    amplitude = np.abs(analytic)
    phase = np.empty((3, 256))
    mapping = (0, 1, 3)
    for r in range(3):
        values = analytic[mapping[r]]
        previous = np.arctan2(values[0].imag, values[0].real)
        correction = 0.
        phase[r, 0] = previous
        for t in range(1, 256):
            angle = np.arctan2(values[t].imag, values[t].real)
            delta = angle-previous
            if delta > np.pi:
                correction -= 2*np.pi
            elif delta < -np.pi:
                correction += 2*np.pi
            phase[r, t] = angle+correction
            previous = angle
        phase[r] -= _mean(phase[r])
    pad = np.zeros((5, 256))
    pad[0], pad[1], pad[4] = phase[0], phase[1], phase[2]
    amp = np.empty((3, 256))
    for r in range(3):
        amp[r] = amplitude[mapping[r]]/(_mean(amplitude[mapping[r]])+1e-9)
    amp_pad = np.zeros((5, 256))
    amp_pad[0], amp_pad[1], amp_pad[4] = amp[0], amp[1], amp[2]
    amp_s3 = amplitude[2]/(_mean(amplitude[2])+1e-9)
    return pad, amp_pad, amp_s3


@njit(cache=True)
def finish_phase(pad, amp_pad, amp_s3, slopes5, amp_slopes5, signals, out, mode):
    phase = np.empty((3, 256))
    amp = np.empty((3, 256))
    for r in range(3):
        index = r if r<2 else 4
        phase[r] = pad[index]
        amp[r] = amp_pad[index]
    slope = np.array([slopes5[0], slopes5[1], slopes5[4]])
    amp_slopes = np.array([amp_slopes5[0], amp_slopes5[1], amp_slopes5[4]])
    centered_s3 = amp_s3-1
    energy_s3 = _mean(centered_s3**2)+1e-9
    lags = np.empty((3, 6))
    amp_lag_stats(amp, centered_s3, energy_s3, lags)
    summaries = np.empty((3, 9))
    summary_stats(phase, slope, amp, summaries)
    for r in range(3):
        offset = 222+(r if r < 2 else 4)*39
        out[offset] = slope[r]
        out[offset+1:offset+9] = summaries[r, :8]
        out[offset+9] = _mean(amp[r]**3)
        out[offset+10] = summaries[r, 8]
        out[offset+11] = amp_slopes[r]
        out[offset+12:offset+17] = lags[r, :5]
        out[offset+38] = lags[r, 5]
    matrix = np.zeros((21, 512))
    if mode == 2:
        kept = np.empty((3, 256))
        kept[0], kept[1], kept[2] = signals[0], signals[1], signals[4]
        windowed_segments(kept, matrix)
    return matrix


@njit(cache=True)
def finish_segments(power, out):
    for r in range(21):
        p = power[r]
        p[0] = 0
        p /= _pairwise_sum(p)+1e-12
        c = r//7
        offset = 222+(c if c < 2 else 4)*39+17+(r%7)*3
        out[offset] = np.argmax(p)/512
        out[offset+1] = np.max(p)
        out[offset+2] = -_pairwise_sum(p*np.log(p+1e-12))


def features(series, mode=2):
    out, z, signals = prepare(series, mode)
    finish_spectral(np.abs(np.fft.rfft(z, axis=1))**2, out)
    if mode == 0:
        return out
    kept = signals[[0, 1, 3, 4]]
    spectrum = np.fft.fft(kept, axis=1)
    spectrum[:, 0] = 0
    spectrum[:, 128:] = 0
    spectrum[:, 1:128] *= 2
    analytic = np.fft.ifft(spectrum, axis=1)
    pad, amp_pad, amp_s3 = phase_inputs(analytic)
    # NumPy performs these two BLAS calls. Keeping @ outside compiled functions
    # avoids Numba's optional SciPy BLAS dependency in the submission environment.
    slopes5 = pad @ TIME / TIME_ENERGY
    amp_slopes5 = amp_pad @ TIME / TIME_ENERGY
    matrix = finish_phase(pad, amp_pad, amp_s3, slopes5, amp_slopes5, signals, out, mode)
    if mode == 2:
        finish_segments(np.abs(np.fft.rfft(matrix, axis=1))**2, out)
    return out
