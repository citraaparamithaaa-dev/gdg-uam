"""Within-signal frequency and amplitude changes, plus compact shape summaries."""
import numpy as np
from signal_features import features as original_features, basic_shape_indices, fill_missing

BASE_INDICES = basic_shape_indices()
TIME = np.arange(256, dtype=float) - 127.5
TIME_ENERGY = np.dot(TIME, TIME)


def extra_features(series):
    raw = fill_missing(series)
    z = (raw - raw.mean(axis=1, keepdims=True)) / (raw.std(axis=1, keepdims=True) + 1e-9)
    signals = np.concatenate([z, (z[0:1] + z[1:2]) / 2], axis=0)
    # Clip isolated extreme observations before measuring repeating structure.
    med = np.median(signals, axis=1, keepdims=True)
    mad = np.median(np.abs(signals-med), axis=1, keepdims=True) * 1.4826 + 1e-6
    signals = np.clip(signals, med-4*mad, med+4*mad)
    signals -= signals.mean(axis=1, keepdims=True)
    signals /= signals.std(axis=1, keepdims=True) + 1e-9
    spectrum = np.fft.fft(signals, axis=1)
    spectrum[:, 0] = 0
    spectrum[:, 128:] = 0
    spectrum[:, 1:128] *= 2
    analytic = np.fft.ifft(spectrum, axis=1)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=1)
    phase -= phase.mean(axis=1, keepdims=True)
    slope = phase @ TIME / TIME_ENERGY
    residual = phase - slope[:, None] * TIME
    instant = np.diff(phase, axis=1)
    amp = amplitude / (amplitude.mean(axis=1, keepdims=True) + 1e-9)
    summaries = [
        slope, residual.std(axis=1), np.mean(np.abs(residual), axis=1),
        np.std(instant, axis=1), np.median(instant, axis=1),
        np.quantile(instant, .9, axis=1)-np.quantile(instant, .1, axis=1),
        instant[:, :127].mean(axis=1), instant[:, 127:].mean(axis=1),
        amp.std(axis=1), np.mean(amp**3, axis=1),
        np.quantile(amp, .9, axis=1)-np.quantile(amp, .1, axis=1),
        amp @ TIME / TIME_ENERGY,
    ]
    amp_centered = amp - 1
    amp_energy = np.mean(amp_centered**2, axis=1) + 1e-9
    for lag in (1, 4, 16, 32, 64):
        summaries.append(np.mean(amp_centered[:,lag:]*amp_centered[:,:-lag], axis=1)/amp_energy)
    for start, width in ((0,128),(64,128),(128,128),(0,64),(64,64),(128,64),(192,64)):
        segment = signals[:,start:start+width]
        segment = (segment - segment.mean(axis=1,keepdims=True)) * np.hanning(width)
        power = np.abs(np.fft.rfft(segment, n=512, axis=1))**2
        power[:,0] = 0
        power /= power.sum(axis=1, keepdims=True) + 1e-12
        summaries.extend([power.argmax(axis=1)/512,
                          power.max(axis=1),
                          -(power*np.log(power+1e-12)).sum(axis=1)])
    # Correlations of event locations in channels 2 and 3.
    summaries.append(np.mean(amp_centered * amp_centered[3:4], axis=1)/np.sqrt(amp_energy*amp_energy[3]))
    return np.nan_to_num(np.stack(summaries, axis=1).ravel(), nan=0., posinf=0., neginf=0.)


def features(series):
    return np.concatenate([original_features(series)[BASE_INDICES], extra_features(series)])


def extra_features_batch(batch):
    """Vectorized extraction for training; numerically equivalent to one-sample extraction."""
    raw = np.stack([fill_missing(series) for series in batch])
    z = (raw-raw.mean(axis=2,keepdims=True))/(raw.std(axis=2,keepdims=True)+1e-9)
    signals = np.concatenate([z,(z[:,0:1]+z[:,1:2])/2],axis=1)
    med = np.median(signals,axis=2,keepdims=True)
    mad = np.median(np.abs(signals-med),axis=2,keepdims=True)*1.4826+1e-6
    signals = np.clip(signals,med-4*mad,med+4*mad)
    signals -= signals.mean(axis=2,keepdims=True)
    signals /= signals.std(axis=2,keepdims=True)+1e-9
    spectrum = np.fft.fft(signals,axis=2)
    spectrum[:,:,0] = 0
    spectrum[:,:,128:] = 0
    spectrum[:,:,1:128] *= 2
    analytic = np.fft.ifft(spectrum,axis=2)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic),axis=2)
    phase -= phase.mean(axis=2,keepdims=True)
    slope = np.sum(phase*TIME,axis=2)/TIME_ENERGY
    residual = phase-slope[:,:,None]*TIME
    instant = np.diff(phase,axis=2)
    amp = amplitude/(amplitude.mean(axis=2,keepdims=True)+1e-9)
    summaries = [slope,residual.std(axis=2),np.mean(np.abs(residual),axis=2),
        np.std(instant,axis=2),np.median(instant,axis=2),
        np.quantile(instant,.9,axis=2)-np.quantile(instant,.1,axis=2),
        instant[:,:,:127].mean(axis=2),instant[:,:,127:].mean(axis=2),
        amp.std(axis=2),np.mean(amp**3,axis=2),
        np.quantile(amp,.9,axis=2)-np.quantile(amp,.1,axis=2),
        np.sum(amp*TIME,axis=2)/TIME_ENERGY]
    centered = amp-1
    energy = np.mean(centered**2,axis=2)+1e-9
    for lag in (1,4,16,32,64):
        summaries.append(np.mean(centered[:,:,lag:]*centered[:,:,:-lag],axis=2)/energy)
    for start,width in ((0,128),(64,128),(128,128),(0,64),(64,64),(128,64),(192,64)):
        segment=signals[:,:,start:start+width]
        segment=(segment-segment.mean(axis=2,keepdims=True))*np.hanning(width)
        power=np.abs(np.fft.rfft(segment,n=512,axis=2))**2
        power[:,:,0]=0
        power/=power.sum(axis=2,keepdims=True)+1e-12
        summaries.extend([power.argmax(axis=2)/512,power.max(axis=2),
                          -(power*np.log(power+1e-12)).sum(axis=2)])
    summaries.append(np.mean(centered*centered[:,3:4],axis=2)/np.sqrt(energy*energy[:,3:4]))
    return np.nan_to_num(np.stack(summaries,axis=2).reshape(len(batch),-1),nan=0.,posinf=0.,neginf=0.)
