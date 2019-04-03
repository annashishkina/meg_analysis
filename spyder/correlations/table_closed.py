# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:07:46 2019

@author: Anna Shishkina
"""
import mne
from mne.time_frequency import psd_welch
import numpy as np
import pandas as pd

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
# Correlation between two psd vectors (first and second half of base_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
base_closed = correlation(psd1, psd2)
print(base_closed)

#------------------------------------------------------------------------------
# Correlation between two psd vectors (first and second half of front_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
front_closed = correlation(psd1, psd2)
print(front_closed)

#------------------------------------------------------------------------------
# Correlation between two psd vectors (first and second half of back_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
back_closed = correlation(psd1, psd2)
print(back_closed)

#------------------------------------------------------------------------------
# Correlation between two psd vectors (first and second half of left_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
left_closed = correlation(psd1, psd2)
print(left_closed)

#------------------------------------------------------------------------------
# Correlation between two psd vectors (first and second half of right_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
right_closed = correlation(psd1, psd2)
print(right_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (base_closed and front_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=60, tmax=120, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
base_front_closed = correlation(psd1, psd2)
print(base_front_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (base_closed and back_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
base_back_closed = correlation(psd1, psd2)
print(base_back_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (base_closed and left_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
base_left_closed = correlation(psd1, psd2)
print(base_left_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (base_closed and right_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\base_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
base_right_closed = correlation(psd1, psd2)
print(base_right_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (front_closed and back_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
front_back_closed = correlation(psd1, psd2)
print(front_back_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (front_closed and left_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
front_left_closed = correlation(psd1, psd2)
print(front_left_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (front_closed and left_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\front_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
front_right_closed = correlation(psd1, psd2)
print(front_right_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (back_closed and left_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
back_left_closed = correlation(psd1, psd2)
print(back_left_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (back_closed and right_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\back_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
back_right_closed = correlation(psd1, psd2)
print(back_right_closed)

#------------------------------------------------------------------------------
# Correlation between two half of psd vectors (left_closed and right_closed condition)
#------------------------------------------------------------------------------
psd1 = multi_power('head_movement\\190131\\left_closed_raw.fif', tmin=0, tmax=60, 
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
psd2 = multi_power('head_movement\\190131\\right_closed_raw.fif', tmin=60, tmax=120,
             fmin_list=[290, 305.5, 313, 319.5], fmax_list=[294, 308.5, 315, 322.5])
left_right_closed = correlation(psd1, psd2)
print(left_right_closed)


#------------------------------------------------------------------------------
# table with correlation coefficients for closed condition
#------------------------------------------------------------------------------
corr_closed = pd.DataFrame({
        'closed condition' : ['base', 'front', 'back', 'left', 'right'],
        'base' : [base_closed, base_front_closed, base_back_closed, base_left_closed, base_right_closed],
        'front' : [base_front_closed, front_closed, front_back_closed, front_left_closed, front_right_closed],
        'back' : [base_back_closed, front_back_closed, back_closed, back_left_closed, back_right_closed],
        'left': [base_left_closed, front_left_closed, back_left_closed, left_closed, left_right_closed],
        'right': [base_right_closed, front_right_closed, back_right_closed, left_right_closed, right_closed],
        })
print(corr_closed)

#------------------------------------------------------------------------------
# exporting and saving pandas data frame as csv file
#------------------------------------------------------------------------------

corr_closed.to_csv("corr_closed.csv", index=False, encoding='utf8')
csv = pd.read_csv('D:\\Git\\meg_analysis\\spyder\\corr_closed.csv')
print(csv)