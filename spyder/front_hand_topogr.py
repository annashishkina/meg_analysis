# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:07:49 2019

@author: Anna Shishkina
"""
#%%
import mne
import numpy as np

#%%
# download data
raw_front_hand = mne.io.read_raw_fif('head_movement\\190131\\front_hand_raw.fif')

# note bad channels
raw_front_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
#%%
# band-pass filter data
raw_front_hand.load_data().filter(l_freq=285., h_freq=295.)
#%%
#channel type selection
picks = mne.pick_types(raw_front_hand.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
#%%
#power spectrum plot
raw_front_hand.plot_psd(fmin=260, fmax=360, n_fft=2048,
             n_jobs=1, proj=False, color=(0, 0, 1),  picks=picks,
             show=False)
#%%
#create epochs
duration = 1.
events = mne.make_fixed_length_events(raw_front_hand, duration=duration)
#reject = dict(grad=4000e-13, mag=4e-12) 

epochs = mne.Epochs(raw_front_hand, events=events, tmin=0., tmax=120,  
                    picks=picks, baseline=(None, 0),  
                        preload=True) 
#%%
evoked = epochs.average()
#%%
#epochs plot
epochs.plot_psd(fmin=260, fmax=360.)

#%%
# set time instants in seconds (from 50 to 150ms in a step of 10ms)
times = np.arange(0.05, 0.15, 0.01)
# If times is set to None only 10 regularly spaced topographies will be shown

# plot magnetometer data as topomaps
evoked.plot_topomap(times, ch_type='mag', time_unit='s')
