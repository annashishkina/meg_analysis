# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:10:54 2019

@author: Shishkina Anna
"""
#%%
import mne
import matplotlib.pyplot as plt
import numpy as np


#%%
data_path = mne.io.read_raw_fif('head_movement\\190131\\front_hand_raw.fif')

#%%
data_path.info['hpi_results']
#%%
data_path.plot_psd()
