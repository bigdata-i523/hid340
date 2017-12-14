#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 03:27:25 2017

Based on the supervised gradident-based learning algorithm specified by 
Reyes-Galaviz et al. (2017).

"""

import numpy as np

input_count = 8 
hidden_count = 5
output_count = 1

eta = 0.02

data = np.load("record_pairs.npy", encoding="latin1")
data2 = np.nan_to_num(data)

# Quality thresholds
tau_1 = 0.01
tau_2 = 0.99

# The MLP model has 8 inputs and 5 nodes in the hidden layer. It follows 
# the Model 2 setup presented by Reyes-Galaviz et al. Hidden nodes aggregate
# values calculated by running each of 8 comparison functions over 5 data
# fields (date, ISBN, title, publisher, creator/contributor)
r = np.random.randn(input_count, hidden_count)
w = np.random.randn(hidden_count, output_count)

# Sigmoid activation function
def activate(a):
    return 1 / (1 + np.exp(-a))

# Partial derivatives for output layer

# MSE
#def E_total(c, s):
#    if c == 0:
#        return 0.5 * np.sum(tau_1 - s)**2
#    else:
#        return 0.5 * np.sum(tau_2 - s)**2
    
#def dQ_ds(c, s):
#    if c == 0:
#        return -(tau_1 - np.sum(s))
#    else:
#        return -(tau_2 - np.sum(s))             
    
def dQ_ds(c, data):              
    if c == 0:            
        if np.sum(data) > tau_1:
            data = 2 * data
            return data
    else:
        if np.sum(data) < tau_2:
            data = 2 * (data - 1)
            return data
    return data

def ds_dl(s):
    return s * (1 - s)

def dy_dp(y):
    return y * (1 - y)
 
output = []

training = data[0:1960]

# Iterations for training model
for i in range(1000):      
      
    training = np.random.permutation(training)    
    
    for t in training:    
        t2 = np.nan_to_num(t)
        for u in t2:                                     
            c = u[:, :][5][0]
        
            u2 = u[:5, :]
        
            # Net input of hidden layer
            p = np.dot(u2, r)        
            
            # Activation of hidden layer
            y = activate(p)
            
            # Chain of partial derivatives for Model 2               
            
            # Net input of output node
            l = np.dot(y, w)        
            
            # Activation of output node
            s = activate(l)   
            
            dE_o = dQ_ds(c, s)
            
            dE_h = np.dot(dE_o, w.T)
            
            dQ_dw = eta * np.dot(y.T, dE_o * ds_dl(s))
                                 
            dQ_dr = eta * np.dot(u2.T, dE_h * dy_dp(y))        
            
            r -= dQ_dr
            w -= dQ_dw
            
            output.append([np.sum(s), c])
             
    
    result = np.array([output])
                        
        
    f = result
    if len(f) > 0:              
        f2 = result[:, :, 1] == 0
        f3 = result[:, :, 1] == 1
        f4 = result[:, :, 0] > 0.5
        f5 = result[:, :, 0] < 0.5
          
        print(len(f[f2 & f4]), len(f[f3 & f5]))
        
    r_final = r
    w_final = w
        
    output = []
    result = []
    #np.savetxt("training_output.txt", np.array([output]))
    
    
    
    
