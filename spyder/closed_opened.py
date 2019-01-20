# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:16:32 2019

@author: annashishkina
"""

#%%
import mne
import matplotlib.pyplot as plt
#%%
"""download fif-files"""
rest_opened = mne.io.read_raw_fif('meg_data\\rest_opened_raw.fif')
rest_closed = mne.io.read_raw_fif('meg_data\\rest_closed_raw.fif')

#%%
%matplotlib auto
#%%
"""select channels by lasso"""
fig_op, selected_channels_op = rest_opened.plot_sensors(kind='select',ch_type='grad')
fig_cl, selected_channels_cl = rest_closed.plot_sensors(kind='select',ch_type='grad')
#%%
"""get list of selected channels"""
%matplotlib auto

#%%
"""pick selected gradiometers"""
grad_picks_opened = mne.pick_types(rest_opened.info, meg='grad', exclude='bads', selection=selected_channels_op)
grad_picks_closed = mne.pick_types(rest_closed.info, meg='grad', exclude='bads',  selection=selected_channels_cl)

#%%
"""plot the power of specific frequency for opened and closed eyes"""
fig, axes = plt.subplots(1,2)
axes[0].set_ylim(8,40)

axes[0].set_xlabel('frequency (Hz)')
fig_open = mne.viz.plot_raw_psd(rest_opened, picks=grad_picks_opened, fmin=4,fmax=20, ax=axes[0])

axes[1].set_ylim(8,40)
axes[1].set_xlabel('frequency (Hz)')
fig_closed = mne.viz.plot_raw_psd(rest_closed, picks=grad_picks_closed, fmin=4,fmax=20, ax=axes[1])
fig.suptitle('rest_opened VS rest_closed', fontsize=14)



