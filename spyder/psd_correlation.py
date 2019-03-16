# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:35:20 2019

@author: Anna Shishkina
"""


import mne
from mne.time_frequency import psd_welch
import numpy as np

#------------------------------------------------------------------------------
# Power spectral density of MEG sensors (gradiometers) for the fist coil (290-294 Hz range)
#------------------------------------------------------------------------------
def power(fname, tmin, tmax, fmin, fmax, n_fft = 2048):
    raw = mne.io.read_raw_fif(fname)
    raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
    psds, freqs = psd_welch(raw, picks=picks, tmin=tmin, tmax=tmax,
                        fmin=fmin, fmax=fmax)
    psds = 20 * np.log10(psds)
    return psds

#------------------------------------------------------------------------------
# Correlation between two psd vectors (first and second half of base_closed condition)
#------------------------------------------------------------------------------
def correlation(psd1, psd2):
    first = np.ravel(psd1)
    second = np.ravel(psd2)
    result = np.corrcoef(first, second)
    correlation = result[0,1]
    return correlation

psd1 = power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, fmin=290, fmax=294)
psd2 = power('head_movement\\190131\\base_closed_raw.fif', tmin=60, tmax=120, fmin=290, fmax=294)
corr_same = correlation(psd1, psd2)
print(corr_same)

#------------------------------------------------------------------------------
# Correlation between two psd vectors (base_closed condition and front_closed condition)
#------------------------------------------------------------------------------
psd1 = power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, fmin=290, fmax=294)
psd2 = power('head_movement\\190131\\front_closed_raw.fif', tmin=0, tmax=60, fmin=290, fmax=294)
corr_diff = correlation(psd1, psd2)
print(corr_diff)


