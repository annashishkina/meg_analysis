# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:32:49 2019

@author: User
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

#%%
raw_front_closed = mne.io.read_raw_fif('head_movement\\190131\\front_closed_raw.fif')
raw_front_closed.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']

raw_front_hand = mne.io.read_raw_fif('head_movement\\190131\\front_hand_raw.fif')
raw_front_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
#%%
# low-pass filter data
raw_front_hand.load_data().filter(l_freq=0, h_freq=40.)

raw_front_closed.load_data().filter(l_freq=0, h_freq=40.)
#%%
picks_fh = mne.pick_types(raw_front_hand.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
picks_fc = mne.pick_types(raw_front_closed.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')

#%%
duration = 1.
events_fh = mne.make_fixed_length_events(raw_front_hand, duration=duration)
events_fc = mne.make_fixed_length_events(raw_front_closed, duration=duration)
reject = dict(grad=4000e-13, mag=4e-12)

epochs_fh = mne.Epochs(raw_front_hand, events=events_fh, tmin=0., tmax=0.99,  
                    picks=picks_fh, baseline=(None, 0),  
                        preload=True, reject = reject)

epochs_fc = mne.Epochs(raw_front_closed, events=events_fc, tmin=0., tmax=0.99,  
                    picks=picks_fc, baseline=(None, 0),  
                        preload=True, reject = reject) 
#%%
epochs_fh.plot_psd_topomap(ch_type='grad', normalize=True)

epochs_fc.plot_psd_topomap(ch_type='grad', normalize=True)
#%%

fig, (up, down) = plt.subplots(nrows=2, figsize=(10, 5))
epochs_fh.plot_psd_topomap(ch_type='grad', normalize=True, axes=up, show=True)
up.set_title("x1")
fig.suptitle("Overall title")
epochs_fc.plot_psd_topomap(ch_type='grad', normalize=True, axes=down, show=True)
down.set_title("x2")







