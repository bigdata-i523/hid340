#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the supervised gradident-based learning algorithm specified by 
Reyes-Galaviz et al. (2017).

"""

import numpy as np

input_count = 8 
hidden_count = 6
output_count = 1

eta = 0.8

data = np.load("record_pairs.npy")
data2 = np.nan_to_num(data)

# Quality thresholds
tau_1 = 0.4
tau_2 = 0.7

# The MLP model has 8 inputs and 6 nodes in the hidden layer. It follows 
# the Model 2 setup presented by Reyes-Galaviz et al. Hidden nodes aggregate
# values calculated by running each of 8 comparison functions over 6 data
# fields (date, ISBN, title, creator, publisher, contributor)
r = np.random.randn(input_count, hidden_count)
w = np.random.randn(hidden_count, output_count)

# Sigmoid activation function
def activate(a):
    return 1 / (1 + np.exp(-a))

# Partial derivatives for output layer
def Q_tau_1(s):    
    return np.sum(s**2)

def Q_tau_2(s):
    return np.sum((1 - s)**2)

def Q(d, s, tau_1, tau_2):
    if d[6:7, 1:2] == 0:        
        return Q_tau_1(s)
    else:
        return Q_tau_2(s)

def dQ_ds(d, s, tau_1, tau_2):    
    if d[6:7, 1:2] == 0: 
        if np.sum(s) > tau_1:
            return 2 * s
        else:
            return s
    else:
        if np.sum(s) < tau_2:            
            return 2 * (s - 1)        
        else:
            return s

def ds_dl(s):
    return s * (1 - s)

# Partial derivatives for hidden layer
def dy_dp(y):
    return y * (1 - y)

# 150 iterations for training model
for i in range(150):      
    
    for u in data2:
    
        # Net input of hidden layer
        p = np.dot(u[:6, :], r)        
        
        # Activation of hidden layer
        y = activate(p)
        
        # Net input of output node
        l = np.dot(y, w)        
        
        # Activation of output node
        s = activate(l)              
        
        # Chain of partial derivatives for Model 2
        # dQ_ds = dQ_ds(u, s, tau_1, tau_2)
        # ds_dl = ds_dl(s)
        dl_dw = y.T
        dl_dy = w.T
        # dQ_dl = dQ_ds * ds_dl
        # dQ_dy = dQ_dl * dl_dy        
        # dy_dp = dy_dp(y)
        dp_dr = u[:6, :].T
        
        # Back propogate                
        dQ_dw = np.dot(dl_dw, (dQ_ds(u, s, tau_1, tau_2) * ds_dl(s)))        
        dQ_dr = np.dot(dp_dr, (dQ_ds(u, s, tau_1, tau_2) * ds_dl(s) * dl_dy * dy_dp(y)))        
                             
        r -= dQ_dr * eta              
        w -= dQ_dw * eta

        print(np.sum(s), u[6:7, 1:2], u[7:8, 1:2])
        

