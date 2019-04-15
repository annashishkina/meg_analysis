# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:33:02 2019

@author: Anna Shishkina
"""

import mne
import numpy as np
from scipy.linalg import eigh 

#------------------------------------------------------------------------------
# Matrix ch x coil for 5 positions based on eigenvector for eigenvalue maximum 
#------------------------------------------------------------------------------

def coil_position (fname):
    raw = mne.io.read_raw_fif(fname, preload = True)
    raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, exclude='bads')
    vectors = []
    for l_freq, h_freq in zip([290, 305.5, 313, 319.5], [294, 308.5, 315, 322.5]):
        frange = raw.filter(l_freq = l_freq, h_freq = h_freq, picks = picks, filter_length = 2048)
        df = frange.to_data_frame(picks = picks, index = 'time', start = 0, stop = 120)
        matrix = df.transpose()
        result = matrix.dot(matrix.T)
        w, v = eigh(result, eigvals=(197,197))
        vectors.append(v)
    position = np.hstack(vectors)
    return np.array(position)

base = coil_position('head_movement\\190131\\base_opened_raw.fif')
front = coil_position('head_movement\\190131\\front_opened_raw.fif')
back = coil_position('head_movement\\190131\\back_opened_raw.fif')
left = coil_position('head_movement\\190131\\left_opened_raw.fif')
right = coil_position('head_movement\\190131\\right_opened_raw.fif')

