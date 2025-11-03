"""ECG transforms and preprocessing
"""
from functools import partial
import numpy as np
from scipy import signal

import torch
from torchvision import transforms as tv_transforms


class ImputeMissingWaveformValues(torch.nn.Module):
    def __init__(self, nan_value):
        super().__init__()
        self.nan_value = nan_value
                 
    def forward(self, X):
        return torch.nan_to_num(X, nan=self.nan_value)


class Scale1D(torch.nn.Module):
    """Scale 1D predictors.
    
    """
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, X):
        dtype = X.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=X.device)
        mean = mean.view(-1, 1)
        std = torch.as_tensor(self.std, dtype=dtype, device=X.device)
        std = std.view(-1, 1)
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to "
                             f"{dtype}, leading to division by zero.")
        return X.sub_(mean).div_(std)


class Identity(torch.nn.Identity):
    def __init__(self):
        super().__init__()
        
    def unscale(self, y):
        return y
    

class ScaleLabel1D(torch.nn.Module):
    """Scale a continuous label.
    """    
    def __init__(self, location, scale):
        super().__init__()
        self.location = location
        self.scale = scale

    def unscale(self, y):
        return (self.scale * y) + self.location
    
    def forward(self, y):
        y = y.astype(np.float64)
        y -= self.location
        y /= (self.scale + 1e-7)
        return y


class CodePatientSex(object):
    def __call__(self, name):
        return [int(name.lower() == 'female'), int(name.lower() == 'male')]


def get_bandpass_filter(
    sampling_rate: float, # sampling rate of `signal` Hz
    bandpass_freq: tuple # 2-tuple of floats, (lower,upper) freq of filter Hz
    )-> tuple:   # 2-tuple filter coefficients (b, a) 
    if np.array(bandpass_freq).shape != (2,):
        raise ValueError(f'invalid `bandpass_freq {bandpass_freq}')
    freq = np.array(bandpass_freq, dtype=np.float32)
    srate = float(sampling_rate)
    nfreq = 2. * freq / srate  # normalize freq to Nyquist freq
    order = int(0.3 * sampling_rate)
    order = order + 1 if order % 2 == 0 else order    
    b = signal.firwin(numtaps=order, cutoff=nfreq, pass_zero=False)
    a = np.array([1])
    return (b, a)


def get_notch_filter(
    sampling_rate: float, # sampling rate of `signal` Hz
    notch_freq: float=60 # freq of filter Hz
    )-> tuple:   # 2-tuple filter coefficients (b, a)     
    srate = float(sampling_rate)
    freq = float(notch_freq)
    b, a = signal.iirnotch(freq, Q=freq/2., fs=srate)
    return (b, a)


def filter_waveform(
    wf: np.ndarray, # original multi-channel signal to be filtered, with shape (Nsamples, Nchannels)
    filter_coef: tuple,  # filter coefficients (b, a)
    axis=0 # axis to apply filter
    )-> np.ndarray:  # filtered version of input `signal`
    b, a = filter_coef
    wf_filtered = signal.filtfilt(b, a, wf, axis=axis)
    return wf_filtered


class EcgFilter(torch.nn.Module):
    """Removal of baseline wander and high frequency noise.
    
    For powerline interference, use a notch filter with `notch_freq` 
    (Default 60 Hz is appropriate for North America; use 50 Hz for Europe)
    For baseline wander and denoising, use a bandpass filter with `bandpass_freq`.
    """
    def __init__(self, sampling_freq=500, notch_freq=60, 
                 bandpass_freq=(0.01, 150), axis=1):
        super().__init__()
        self.sampling_freq = sampling_freq
        self.notch_freq = notch_freq
        self.bandpass_freq = bandpass_freq
        self.axis = axis
        self.set_filters(sampling_freq)
    
    def set_filters(self, sampling_freq):
        bp_coef = get_bandpass_filter(sampling_freq, self.bandpass_freq)
        notch_coef = get_notch_filter(sampling_freq, self.notch_freq)
        self.filter_1 = partial(filter_waveform, filter_coef=notch_coef,
                                axis=self.axis)
        self.filter_2 = partial(filter_waveform, filter_coef=bp_coef, 
                                axis=self.axis)
                    
    def set_sampling_freq(self, f):
        if f != self.sampling_freq:
            self.set_filters(f)

    def forward(self, X):
        X_filtered = self.filter_2([self.filter_1(it) for it in X]).copy()
        X_filtered = torch.tensor(X_filtered).float().to(X.device)
        return X_filtered


def create_transforms(dataset, input_size):
    """Return a transform dict key'd to `dataset.xkey_name`.
    For continuous scalar targets, an extra key 'y' will provide a list of 
    transforms (for those with do_scale == True).
    For continous scalar covariates, an extra key 'covar' will provide a list
    of transforms.
    """
    tfd = {}
    d = dataset.data
    if len(input_size) == 2:
        in_chans, _ = input_size
    else:
        _, in_chans, _ = input_size
    for key in d.xkey_name:
        mean_val = d.mean[key]
        mean_tup = tuple([mean_val] * in_chans)
        std_tup = tuple([d.std[key]] * in_chans)
        tf_list = [
            ImputeMissingWaveformValues(nan_value=mean_val),
        ]
        if d.ecg_filter:
            tf_list += [EcgFilter(notch_freq=d.notch_freq,
                                  bandpass_freq=d.bandpass_freq)]
        tf_list += [Scale1D(mean=mean_tup, std=std_tup)]
        tfd[key] = tv_transforms.Compose(tf_list)
        
    if d.scale_y:
        if len(d.ykey) == 1:
            tfd['y'] = ScaleLabel1D(d.y_train_loc, d.y_train_scale)
        else:
            ts = []
            for i, do_scale in enumerate(d.ykey_scale):
                if do_scale:
                    ts.append(ScaleLabel1D(d.y_train_loc[i], 
                                           d.y_train_scale[i]))
                else:
                    ts.append(Identity())
            tfd['y'] = ts
    return tfd
