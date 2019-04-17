# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 19:35:20 2019

@author: Anna Shishkina
"""


import mne
from mne.time_frequency import psd_welch
import numpy as np

#------------------------------------------------------------------------------
# Power spectral density of MEG sensors (gradiometers) 
#------------------------------------------------------------------------------
def power(fname, tmin, tmax, fmin, fmax, n_fft = 2048):
    raw = mne.io.read_raw_fif(fname)
    raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
    psds, freqs = psd_welch(raw, picks=picks, tmin=tmin, tmax=tmax,
                        fmin=fmin, fmax=fmax, n_fft = n_fft)
    psds = psds.mean(1)
    return psds

#------------------------------------------------------------------------------
# Correlation between two psd vectors 
#------------------------------------------------------------------------------
def correlation(psd1, psd2):
    result = np.corrcoef(psd1.flatten(), psd2.flatten())
    correlation = result[0,1]
    return correlation

#------------------------------------------------------------------------------
# Power spectral density of MEG sensors (gradiometers) for the coils (4 frequency ranges)
#------------------------------------------------------------------------------
def multi_power (fname, tmin, tmax, fmin_list, fmax_list):
    multi = []
    for fmin, fmax in zip(fmin_list, fmax_list):
        psd = power(fname, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        multi.append(psd)
    return np.array(multi)

      
#------------------------------------------------------------------------------
# Correlation between first and second half of base_closed condition for 4 coils
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
                   fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=60, tmax=120, 
                   fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
corr_same = correlation(psd1, psd2)
print(corr_same)

raw = mne.io.read_raw_fif('head_movement\\190131\\move_closed_raw.fif')
raw.plot_psd()
