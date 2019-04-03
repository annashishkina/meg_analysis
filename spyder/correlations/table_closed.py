# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:07:46 2019

@author: Anna Shishkina
"""
import pandas as pd

#------------------------------------------------------------------------------
# table with correlation coefficients for closed condition
#------------------------------------------------------------------------------
corr_closed = pd.DataFrame({
        'closed condition' : ['base', 'front', 'back', 'left', 'right'],
        'base' : [base_closed, base_front_closed, base_back_closed, base_left_closed, base_right_closed],
        'front' : [base_front_closed, front_closed, front_back_closed, front_left_closed, front_right_closed],
        'back' : [base_back_closed, front_back_closed, back_closed, back_left_closed, back_right_closed],
        'left': [base_left_closed, front_left_closed, back_left_closed, left_closed, left_right_closed],
        'right': [base_right_closed, front_right_closed, back_right_closed, left_right_closed, right_closed],
        })
print(corr_closed)