#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:18:46 2017

@author: tat2
"""

import numpy as np

def activate(a):
    return 1 / (1 + np.exp(-a))

input_count = 8 #+ 1 # Bias node
hidden_count = 5 #+ 1 # Bias node
output_count = 1

eta = 0.8

data = np.load("record_pairs.npy")
data2 = np.nan_to_num(data)


#np.array([[0.12, 0.02, 1, 0], [0.02, 0.16, 1, 0], [0.12, 0.26, 1, 0], [0.92, 0.88, 1, 1]])
#label = 1
tau_1 = 0.4
tau_2 = 0.7

r = np.random.randn(input_count, hidden_count)
w = np.random.randn(hidden_count, output_count)

# Partial derivatives for output layer
def Q_tau_1(s):    
    return np.sum(s**2)

def Q_tau_2(s):
    return np.sum((1 - s)**2)

def Q(d, s, tau_1, tau_2):
    if d[5:6, 1:2] == 0:        
        return Q_tau_1(s)
    else:
        return Q_tau_2(s)

def dQ_ds(d, s, tau_1, tau_2):    
    if d[5:6, 1:2] == 0:        
        return 2 * s
    else:
        return 2 * (s - 1)        

def ds_dl(s):
    return s * (1 - s)

# Partial derivatives for hidden layer
def dy_dp(y):
    return y * (1 - y)

def dp_dr(u):
    return u

for i in range(150):      
    
    for u in data2:
        
        p = np.dot(u[:5, :], r)        
        
        y = activate(p)
        
        l = np.dot(y, w)        
        
        s = activate(l)              
    
        p1 = dQ_ds(u, s, tau_1, tau_2)        
        p2 = ds_dl(s)
        p3a = y.T
        p3b = w.T
        p4 = dy_dp(y)
        p5 = u[:5, :].T
        
        dQ_dw = np.dot(p3a, (p1 * p2))        
        dQ_dr = np.dot(p5, (p1 * p3b * p4))        
                             
        r -= dQ_dr * eta      
        
        w -= dQ_dw * eta

        #print("r: ", r)
        #print("w: ", w)
        print(Q(u, s, tau_1, tau_2), u[5:6, 1:2])
        #print(np.sum(s), u[5:6, 1:2])
        

