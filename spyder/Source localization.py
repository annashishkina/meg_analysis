# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:03:22 2019

@author: User
"""
#%%
import mne
import numpy as np
from os import path as op
from mne.datasets import sample

#%%
#read data from empty room and our data for hand movement
raw_empty_room = mne.io.read_raw_fif('head_movement\\190131\\background_before_raw.fif')
raw_front_hand = mne.io.read_raw_fif('head_movement\\190131\\front_hand_raw.fif')
raw_front_hand.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']

#make sure bad channels get stored in the covariance object
raw_empty_room.info['bads'] = raw_front_hand.info['bads']
raw_empty_room.add_proj([pp.copy() for pp in raw_front_hand.info['projs']])

#%%
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
#%%
#compute covariance matrix
#fname_cov = mne.compute_raw_covariance(
 #   raw_empty_room, tmin=0, tmax=None)
#%%
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
#%%
# The events are spaced evenly every 1 second.
duration = 1.

# create a fixed size events array
# start=0 and stop=None by default
events = mne.make_fixed_length_events(raw_front_hand, duration=duration)

# for fixed size events no start time before and after event
tmin = 0.
tmax = 0.99  # inclusive tmax, 1 second epochs

# Epoched data
epochs = mne.Epochs(raw_front_hand, events=events, tmin=tmin,
                    tmax=tmax, baseline=None, verbose=True)
epochs.drop_bad()
#%%
#epochs.plot(scalings='auto', block=True)
#%%
# Evoked data
nave = len(epochs)  # Number of averaged epochs
evoked_data = np.mean(epochs, axis=0)
fname_ave = mne.Evoked(, condition=None, proj=True, kind='average', allow_maxshield=False, verbose=None)
fname_ave = mne.EvokedArray(evoked_data, tmin=-0.2, info=raw_front_hand.info,
                          comment='Arbitrary', nave=nave)
#%%
#fname_ave.plot(show=True, units={'mag': '-'},
#             titles={'mag': 'sin and cos averaged'}, time_unit='s')
#%%

data_path = sample.data_path() #path="D:\Git\meg_analysis\spyder\mne_datasets", update_path = True)
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'

#%%
data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
#%%
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                          baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)
evoked_full = evoked.copy()
evoked.crop(0.07, 0.08)
#%%

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]
#%%

# Plot the result in 3D brain with the MRI image.
dip.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview')
#%%






