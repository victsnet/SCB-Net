#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:43:10 2023

@author: silva
"""
import numpy as np

def calc_weights(prob_mask, ths=0.5):
    
    dim = prob_mask.shape
    loc = np.argmin(dim)
    
    weights = []
    proportions = []
    n_total = np.sum(np.where(np.max(prob_mask, -1) > 0.0, 1.0, 0.0))
    for n in range(prob_mask.shape[loc]):
        
        if loc == 0:
            n_samples_class = np.sum(np.where(prob_mask[:, :, n] > 0.0, 1.0, 0.0))
            n_samples_ths = np.sum(np.where(prob_mask[:, :, n] >= ths, 1.0, 0.0))
            
            weights.append(n_total/(n_samples_class * dim[loc] + 1e-8))
            proportions.append(n_samples_class/(n_total + 1e-8))
            
        if loc == 2:
            n_samples_class = np.sum(np.where(prob_mask[:, :, n] > 0.0, 1.0, 0.0))
            n_samples_ths = np.sum(np.where(prob_mask[:, :, n] > ths, 1.0, 0.0))
            
            weights.append(n_total/(n_samples_class * dim[loc] + 1e-8))
            proportions.append(n_samples_class/(n_total + 1e-8))
            
    proportions = np.array(proportions)
    proportions = np.expand_dims(np.expand_dims(np.expand_dims(proportions, axis=0), axis=0), axis=0)
    proportions = np.float32(proportions)
    weights = np.array(weights)
    weights = np.expand_dims(np.expand_dims(np.expand_dims(weights, axis=0), axis=0), axis=0)
    weights = np.float32(weights)
    
    return weights, proportions
