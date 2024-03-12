#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:06:54 2023

@author: silva
"""

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral, unary_from_softmax
from tqdm import tqdm
import tensorflow as tf
from scipy.signal import convolve2d
#import segmentation_models as sm


class predict_tiles:
    
    def __init__(self, model, merge_func=np.nanmean, add_padding=False, reflect=False, pad=8):
        
        self.model = model
        self.add_padding = add_padding
        self.reflect = reflect
        self.merge_func = merge_func
        self.bayesian = False
        self.pad = pad
        
    def bayesian_predict_knuth(self, model, x, n_samples=10):
        """
        Predict mean and variance of output given input `x` using the Knuth method.
        
        :param x: Input tensor of shape `(batch_size, height, width, channels)`.
        :param n_samples: Number of Monte Carlo samples to draw during prediction.
        
        :return: Tuple of mean and variance tensors, each of shape `(batch_size, height, width, channels)`.
        """
        # Initialize variables
        mean = tf.zeros_like(model.predict(x, verbose=0))
        var = tf.zeros_like(model.predict(x, verbose=0))

        # Compute mean and variance incrementally
        for i in range(n_samples):
            y = model.predict(x, verbose=0)
            delta = y - mean
            mean += delta / (i + 1)
            var += delta * (y - mean)

        var /= (n_samples - 1)

        return mean.numpy(), var.numpy()
        
    def bayesian_pred(self, model, x, n_samples=10, verbose=0, spatial_model=False):
        
        # Perform Bayesian predictions with Monte Carlo dropout
        if spatial_model:
            y_preds = np.full((n_samples, *x[0].shape[:-1], self.n_classes), -9999.)
        else:
            y_preds = np.full((n_samples, *x.shape[:-1], self.n_classes), -9999.)
            
        for i in range(n_samples):
            if spatial_model:
                y_preds[i], _ = model.predict(x, verbose=verbose)
            else:
                y_preds[i], _ = model.predict(x, verbose=verbose)
        
        # Compute the mean and standard deviation of the predicted probabilities
        y_mean = y_preds.mean(axis=0)
        y_std = y_preds.var(axis=0)
        del y_preds
        
        return y_mean, y_std
        
        
    def overlap(self, im1, im2, positional=0):
        
        if positional == 0:
            im1[:self.pad, :] = np.nan; im1[-self.pad:, :] = np.nan
            im1[:, -self.pad:] = np.nan; im1[:, :self.pad] = np.nan
            
        im2[:self.pad, :] = np.nan; im2[-self.pad:, :] = np.nan
        im2[:, -self.pad:] = np.nan; im2[:, :self.pad] = np.nan
            
            
        result = self.merge_func([im1, im2], 0)            
        return result
        
        
    def create_batches(self, data, dim, overlap_ratio, n_classes):
        
        if self.add_padding:
            data = cv2.copyMakeBorder(data, dim[0]//2, dim[1]//2, dim[0]//2, dim[1]//2, cv2.BORDER_REFLECT)
            
            
        step = int((1-overlap_ratio)*dim[0])    
        (self.y_max, self.x_max, _) = data.shape
        sy = self.y_max//step; sx = self.x_max//step
        batch             = np.zeros((sy*sx, *dim))
        self.dim          = dim
        self.step         = step
        self.n_classes    = n_classes
        
        if self.reflect:
            batch  = np.full((4*sy*sx, *dim), np.nan)            

        n = 0
        for y in range(dim[1]//2, self.y_max-dim[1]//2, self.step):
            for x in range(dim[0]//2, self.x_max-dim[0]//2, self.step):
                batch[n] = data[y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2, :]
                n += 1
                
        if self.reflect:
            m = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2, -self.step):
                    batch[n+m] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    m += 1 
                
            j = 0
            for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2, -self.step):
                    batch[n+m+j] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    j += 1 
                    
                    
            k = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                    batch[n+m+j+k] = data[y-self.dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :]
                    k += 1 
                    
        self.batches = batch
        self.num = n
        del batch
        
        if self.reflect:
            self.num = n+m+j+k
                        
    def predict(self, batches_num, extra_channels=0, output=0, pad=8):
        
        results = []
        
        for n in tqdm(range(0, self.num, batches_num)):
            
            if extra_channels > 0:
                p, p0 = self.model.predict([self.batches[:batches_num, :, :, :extra_channels], 
                                    self.batches[:batches_num, :, :, extra_channels:]], spatial_model=True, verbose=0)
            else: 
                p, p0 = self.model.predict(self.batches[:batches_num], verbose=0)
                
            if output == 1:
                p = p0
                
            results.append(p)
            self.batches = self.batches[batches_num:]
            
        self.results = np.concatenate(results)
        del self.batches
        del results
        
    def bayesian_prediction(self, batches_num, extra_channels=0, n_samples=10, pad=8):
        
        results_mean = []
        results_var = []
        
        for n in tqdm(range(0, self.num, batches_num)):
            
            if extra_channels > 0:
                p_mean, p_var = self.bayesian_pred(self.model, ([self.batches[:batches_num, :, :, :extra_channels], 
                                    self.batches[:batches_num, :, :, extra_channels:]]), n_samples, verbose=0, spatial_model=True)
                
                
            else: 
                p_mean, p_var = self.bayesian_pred(self.model, self.batches[:batches_num], n_samples)
            
            
            results_mean.append(p_mean)
            results_var.append(p_var)
            self.batches = self.batches[batches_num:]
            
        self.results_mean = np.concatenate(results_mean)
        self.results_var = np.concatenate(results_var)
        self.bayesian = True
        del self.batches
        del results_var
        del results_mean
        
    def reconstruct(self, results):
        # reserve memory
        grid = np.full((1, self.y_max, self.x_max, self.n_classes), np.nan)

        n = 0
        for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
            for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                grid[:, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                results[n], -1)

                n += 1     
        
        if self.reflect:
            m = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                     self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m], -1)
                    m += 1 
                        
            j = 0
            for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m+j], -1)
                    j += 1 
                        
                        
            k = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m+j+k], -1)
                    k += 1 
                    
        if self.add_padding:
            return grid[0, self.dim[1]//2:-self.dim[1]//2, self.dim[0]//2:-self.dim[0]//2, :]
            
        else: return grid[0]
    
          
    def merge(self):
        
        if self.bayesian:
            output_mean = np.float32(self.reconstruct(self.results_mean))
            output_var = np.float32(self.reconstruct(self.results_var))
            
            return output_mean, output_var
            
        else:
            output = np.float32(self.reconstruct(self.results))
            
            return output
        
        
def dense_crf(image, probabilities, compat=3, gw=11, bw=3, sch=7, n_iterations=5):
        
    '''
    gw - pairwise gaussian window size: enforces more spatially consistent segmentations.
    bw - pairwise bilateral window size: uses local color features to refine predictions.
    '''
    
    ny = image.shape[0]
    nx = image.shape[1]
    n_classes = probabilities.shape[-1]
    softmax = probabilities.squeeze()
    softmax = softmax.transpose((2, 0, 1))

	# The input should be the negative of the logarithm of probability values
	# Look up the definition of the unary_from_softmax for more information
    unary = unary_from_softmax(softmax, scale=None, clip=1e-5)

	# The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(ny*nx, n_classes)

    d.setUnaryEnergy(unary)

	# This potential penalizes small pieces of segmentation that are
	# spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(gw, gw), shape=(ny, nx))

    d.addPairwiseEnergy(feats, compat=compat,
                    	kernel=dcrf.DIAG_KERNEL,
                    	normalization=dcrf.NORMALIZE_SYMMETRIC)

	# This creates the color-dependent features --
	# because the segmentation that we get from CNN are too coarse
	# and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(bw, bw), schan=(sch, sch, sch),
									img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=compat,
                     	kernel=dcrf.DIAG_KERNEL,
                     	normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(n_iterations)
    probs = np.array(Q, dtype=np.float32).reshape((n_classes, ny, nx))

    return probs.swapaxes(1, 0).swapaxes(1, 2)

def create_kernel(N, A, B):
    kernel = np.full((N, N), B)
    center = N // 2
    kernel[center, center] = A
    return kernel

def spatial_constraint(input_matrix, reference, constraint_strength=0.9, threshold=0., kernel_size=3):
    
    """
    Apply a kernel to an image at certain central locations.

    Parameters:
    image (numpy.ndarray): The input image.
    reference (numpy.ndarray): The input mask for calculating the locations.
    kernel (numpy.ndarray): The kernel to apply.
    locations (list): A list of tuples representing the central locations where the kernel will be applied.

    Returns:
    numpy.ndarray: The output image.
    """
    # make sure the array is float
    input_matrix = np.float32(input_matrix)
    # surroundings
    surr = (1.-constraint_strength + 1e-7)/(kernel_size**2-1) 
    kernel = create_kernel(kernel_size, constraint_strength, surr)
    pad = (2 * kernel_size - 1)//2 
    
    # Pad the array with 1 row and 1 column of zeros on each side
    image = np.pad(input_matrix, ((pad, pad), (pad, pad)), mode='constant')
    reference = np.pad(reference, ((pad, pad), (pad, pad)), mode='constant')
    locy, locx = np.where(reference > threshold)[:2]
    
    # Create an empty output image
    output_image = image.copy()
    
    # Iterate through the list of locations
    for row, col in zip(locy, locx):        
        # Apply the kernel to the subimage around the central location
        subimage = image[row-pad:row+pad+1, col-pad:col+pad+1]
        suboutput = convolve2d(subimage, kernel, mode='same')
        # Assign the output to the corresponding region in the output image
        output_image[row-1:row+2, col-1:col+2] =  np.float32(suboutput)[pad:-pad, pad:-pad]
    
    return output_image[pad:-pad, pad:-pad]