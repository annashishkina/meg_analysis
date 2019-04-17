# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:33:02 2019

@author: Anna Shishkina
"""

import mne
import numpy as np
from scipy.linalg import eig 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#------------------------------------------------------------------------------
# Matrix ch x coil for 5 positions based on eigenvector for eigenvalue maximum 
#------------------------------------------------------------------------------

def coil_position (fname):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.info['bads'] += ['MEG1433','MEG1822','MEG1843','MEG1412','MEG0943','MEG1033']
    raw.info['sfreq'] = 648.
    picks = mne.pick_types(raw.info, meg='grad', eeg=False, eog=False,
                       stim=False, exclude='bads')
    raw.crop(0.,120.)
    vectors = []
    for l_freq, h_freq in zip([290, 305.5, 313, 319.5], [294, 308.5, 315, 322.5]):
        raw_copy = raw.copy()
        frange = raw_copy.filter(l_freq=l_freq, h_freq=h_freq, picks=picks)
        ch_time = frange.get_data(picks=picks, start = 0, stop = 120001)
        matrix = ch_time.dot(ch_time.T)
        w, v = eig(matrix)
        princ = v[:1]
        vectors.append(princ)
    position = np.hstack(vectors)
    position.reshape
    return np.array(position)
  
a = front.reshape(4,198)

base = coil_position('head_movement\\190131\\base_opened_raw.fif').reshape(4,198).transpose()
front = coil_position('head_movement\\190131\\front_opened_raw.fif').reshape(4,198).transpose()
back = coil_position('head_movement\\190131\\back_opened_raw.fif').reshape(4,198).transpose()
left = coil_position('head_movement\\190131\\left_opened_raw.fif').reshape(4,198).transpose()
right = coil_position('head_movement\\190131\\right_opened_raw.fif').reshape(4,198).transpose()


#------------------------------------------------------------------------------
# Visualization part
#------------------------------------------------------------------------------


fig, ax = plt.subplots(1, 5)
axes = np.array((ax[0], ax[1], ax[2], ax[3], ax[4]))

c = ax[0].pcolor(base,  cmap = 'viridis')
ax[0].set_title('base')
c = ax[1].pcolor(front,  cmap = 'viridis')
ax[1].set_title('front')
c = ax[2].pcolor(back,  cmap = 'viridis')
ax[2].set_title('back')
c = ax[3].pcolor(left, cmap = 'viridis')
ax[3].set_title('left')
c = ax[4].pcolor(right, cmap = 'viridis')
ax[4].set_title('right')


for pos in ([0, 1, 2, 3, 4]):
    ax[pos].xaxis.set_major_formatter(ticker.NullFormatter())
    ax[pos].xaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5,3.5]))
    ax[pos].xaxis.set_minor_formatter(ticker.FixedFormatter(['1','2','3','4']))
    
fig.tight_layout()
fig.colorbar(c, ax=axes.ravel().tolist())
plt.show()
plt.savefig('head_pos_eig.png')

