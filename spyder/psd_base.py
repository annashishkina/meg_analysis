# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:35:22 2019

@author: 
"""


import mne
from mne.time_frequency import psd_welch
import numpy as np

#READ MEG DATA FOR BASE CLOSED CONDITION DURING THE FIRST HALF OF DATA
raw = mne.io.read_raw_fif('head_movement\\190131\\base_closed_raw.fif')
raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
#raw.plot_psd(picks=picks, tmin=tmin, tmax=tmax)
tmin, tmax = 0, 60  # use the first 120s of data
fmin, fmax = 290, 294  # look at frequencies between 
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
psds, freqs = psd_welch(raw, picks=picks, tmin=tmin, tmax=tmax,
                        fmin=fmin, fmax=fmax)
psds = 20 * np.log10(psds)

#READ MEG DATA FOR BASE CLOSED CONDITION DURING THE FIRST HALF OF DATA
raw2 = mne.io.read_raw_fif('head_movement\\190131\\base_closed_raw.fif')
raw2.info
raw2.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
picks2 = mne.pick_types(raw2.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
#raw.plot_psd(picks=picks, tmin=tmin, tmax=tmax)
tmin2, tmax2 = 60, 120  # use the first 120s of data
fmin, fmax = 290, 294  # look at frequencies between 
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
psds2, freqs2 = psd_welch(raw2, picks=picks2, tmin=tmin2, tmax=tmax2,
                        fmin=fmin, fmax=fmax)
psds2 = 20 * np.log10(psds2)

#CALCULATE THE CORRELATION BETWEEN CHANNELS POWER 
first_half = np.ravel(psds)
second_half = np.ravel(psds2)
result1 = np.corrcoef(first_half, second_half)
corr1 = result1[0,1]
print(corr1)

#READ MEG DATA FOR FRONT CLOSED CONDITION DURING THE FIRST HALF OF DATA
raw3 = mne.io.read_raw_fif('head_movement\\190131\\front_closed_raw.fif')
raw3.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
picks3 = mne.pick_types(raw3.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
#raw3.plot_psd(picks=picks, tmin=tmin, tmax=tmax)
tmin3, tmax3 = 0, 60  # use the first 120s of data
fmin, fmax = 290, 294  # look at frequencies between 
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
psds3, freqs3 = psd_welch(raw3, picks=picks3, tmin=tmin3, tmax=tmax3,
                        fmin=fmin, fmax=fmax)
psds3 = 20 * np.log10(psds3)

#CALCULATE THE CORRELATION BETWEEN CHANNELS POWER 
base = np.ravel(psds)
front = np.ravel(psds3)
result2 = np.corrcoef(base, front)
corr2 = result2[0,1]
print(corr2)

#READ MEG DATA FOR MOVE CLOSED CONDITION DURING THE FIRST HALF OF DATA
raw4 = mne.io.read_raw_fif('head_movement\\190131\\base_closed_raw.fif')
raw4.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
picks4 = mne.pick_types(raw4.info, meg='grad', eeg=False, eog=False,
                           stim=False, exclude='bads')
#raw3.plot_psd(picks=picks, tmin=tmin, tmax=tmax)
tmin4, tmax4 = 0, 60  # use the first 120s of data
fmin, fmax = 290, 294  # look at frequencies between 
n_fft = 2048  # the FFT size (n_fft). Ideally a power of 2
psds4, freqs4 = psd_welch(raw4, picks=picks4, tmin=tmin4, tmax=tmax4,
                        fmin=fmin, fmax=fmax)
psds4 = 20 * np.log10(psds4)

#CALCULATE THE CORRELATION BETWEEN CHANNELS POWER 
base = np.ravel(psds)
move = np.ravel(psds4)
result3 = np.corrcoef(base, move)
corr3 = result3[0,1]
print(corr3)


