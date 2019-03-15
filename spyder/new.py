# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:07:49 2019

@author: User
"""
#%%
import mne
import numpy as np
import matplotlib.pyplot as plt


#%%
raw_front_hand = mne.io.read_raw_fif('head_movement\\190131\\front_hand_raw.fif')
raw_front_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
#%%
# low-pass filter data
raw_front_hand.load_data().filter(l_freq=0, h_freq=40.)
#%%
picks_meg = mne.pick_types(raw_front_hand.info, meg=True, eeg=False, eog=False,
                           stim=False, exclude='bads')
#%%
raw_front_hand.plot()
#%%
method = 'fastica'
n_components = 25  # if float, select n_components by explained variance of PCA
random_state = 23
decim = 3
#%%
ica = ICA(n_components=n_components, method=method, random_state = random_state)
#%%
reject = dict(mag=5e-12, grad=4000e-13)
ica.fit(raw_front_hand, picks=picks_meg, decim=decim, reject=reject)
#%%
picks = picks_meg[:4]

#%%
raw_front_hand.plot_psd(tmin=0, tmax=127., fmin=260, fmax=400, n_fft=2048,
             n_jobs=1, proj=False, color=(0, 0, 1),  picks=picks,
             show=False, average=True)
#%%
duration = 1.
events = mne.make_fixed_length_events(raw_front_hand, duration=duration)
reject = dict(grad=4000e-13, mag=4e-12)

epochs = mne.Epochs(raw_front_hand, events=events, tmin=0., tmax=0.99,  
                    picks=picks_meg, baseline=(None, 0),  
                        preload=True, reject = reject) 
#%%
epochs.plot_psd(fmin=2., fmax=40.)

#%%
evoked = epochs.average()  
#%%
evoked.plot()
#%%
# define frequencies of interest (log-spaced)
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
#%%
power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
#%%
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[0],
                   title='Alpha', show=False)
power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='logratio', axes=axis[1],
                   title='Beta', show=False)

#%%
csd_fft = csd_fourier(epochs, fmin=260, fmax=400, n_jobs=1)
#%%
epochs.plot_psd_topomap(ch_type='grad', normalize=True)

