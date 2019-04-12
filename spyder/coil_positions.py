# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:08:32 2019

@author: Anna Shishkina
"""

import mne
import numpy as np

def coil_position (fname):
    raw = mne.io.read_raw_fif(fname, preload = True)
    raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, exclude='bads')
    position = []
    for l_freq, h_freq in zip([290, 305.5, 313, 319.5], [294, 308.5, 315, 322.5]):
        frange = raw.filter(l_freq = l_freq, h_freq = h_freq, picks = picks)
        start, stop = frange.time_as_index([0, 120])
        data, times = frange[picks, start:stop]
        ch_time = np.array(data)
        time_ch = ch_time.transpose()
        matrix = ch_time @ time_ch
        u, s, vh = np.linalg.svd(matrix, full_matrices=True)
        position.append(s)
    return np.array(position)

base = coil_position('head_movement\\190131\\base_opened_raw.fif').transpose()
front = coil_position('head_movement\\190131\\front_opened_raw.fif').transpose()
back = coil_position('head_movement\\190131\\back_opened_raw.fif').transpose()
left = coil_position('head_movement\\190131\\left_opened_raw.fif').transpose()
right = coil_position('head_movement\\190131\\right_opened_raw.fif').transpose()