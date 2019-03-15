# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:35:23 2019

@author: User
"""
import mne
import numpy as np

#%%
# download data
raw_move_hand = mne.io.read_raw_fif('head_movement\\190131\\move_hand_raw.fif')

# note bad channels
raw_move_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
#%%
# band-pass filter data
raw_move_hand.load_data().filter(l_freq=260, h_freq=400.)
#%%
#channel type selection
picks = mne.pick_types(raw_move_hand.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
#%%
#power spectrum plot
raw_move_hand.plot_psd(fmin=260, fmax=360, n_fft=2048,
             n_jobs=1, proj=False, color=(0, 0, 1),  picks=picks,
             show=False)