# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:00:09 2019

@author: Anna Shishkina
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

#%%
# download data
raw_back_hand = mne.io.read_raw_fif('head_movement\\190131\\back_hand_raw.fif')

# note bad channels
raw_back_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
#%%
# band-pass filter data
raw_back_hand.load_data().filter(l_freq=260, h_freq=400.)
#%%
#channel type selection
picks = mne.pick_types(raw_back_hand.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
#%%
#power spectrum plot
raw_back_hand.plot_psd(fmin=260, fmax=360, n_fft=2048,
             n_jobs=1, proj=False, color=(0, 0, 1),  picks=picks,
             show=False)
#%%
#create epochs
duration = 1.
events = mne.make_fixed_length_events(raw_back_hand, duration=duration)
#reject = dict(grad=4000e-13, mag=4e-12) 

epochs = mne.Epochs(raw_back_hand, events=events, tmin=0., tmax=0.99,  
                    picks=picks, baseline=(None, 0),  
                        preload=True) 
#%%
#epochs plot
epochs.plot_psd(fmin=260, fmax=360.)
