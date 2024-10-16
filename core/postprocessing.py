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
                        
    def predict(self, batches_num, extra_channels=0, output=0):
        
        results = []
        
        for n in tqdm(range(0, self.num, batches_num)):
            if extra_channels > 0:
                p, p0 = self.model.predict([self.batches[:batches_num, :, :, :extra_channels], 
                                    self.batches[:batches_num, :, :, extra_channels:]], spatial_model=True, verbose=0)
            else: 
                p, p0 = self.model.predict(self.batches[:batches_num], spatial_model=False, verbose=0)
                
            if output == 1:
                p = p0
                
            results.append(p)
            self.batches = self.batches[batches_num:]
            
        self.results = np.concatenate(results)
        del self.batches
        del results
        
    def bayesian_prediction(self, batches_num, extra_channels=0, n_samples=10):
        
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
        
    def reconstruct(self, results, positional=-1):
        # reserve memory
        grid = np.full((1, self.y_max, self.x_max, self.n_classes), np.nan)

        n = 0
        for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
            for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                grid[:, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                results[n], positional)

                n += 1     
        
        if self.reflect:
            m = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                     self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m], positional)
                    m += 1 
                        
            j = 0
            for y in range(self.dim[1]//2, self.y_max-self.dim[1]//2, self.step):
                for x in range(self.x_max-self.dim[0]//2, self.dim[0]//2,  -self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m+j], positional)
                    j += 1 
                        
                        
            k = 0
            for y in range(self.y_max-self.dim[1]//2, self.dim[1]//2, -self.step):
                for x in range(self.dim[0]//2, self.x_max-self.dim[0]//2, self.step):
                    grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,] =\
                    self.overlap(grid[0, y-self.dim[1]//2:y+self.dim[1]//2, x-self.dim[0]//2:x+self.dim[0]//2,], 
                                    results[n+m+j+k], positional)
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

def dense_crf(image, probs, sxy_gaussian=(5, 5), sxy_bilateral=(80, 80), srgb_bilateral=(13, 13, 13), n_iter=10):
    """
    Applies DenseCRF to the softmax probabilities to refine segmentation results.
    
    Parameters:
    - image: Input image as a numpy array of shape (H, W, C) where C is the number of channels (could be 1, 3, etc.)
    - probs: Softmax output of the CNN, shape (H, W, num_classes)
    - sxy_gaussian: Standard deviation for Gaussian pairwise term
    - sxy_bilateral: Standard deviation for Bilateral pairwise term (spatial coordinates)
    - srgb_bilateral: Standard deviation for Bilateral pairwise term (RGB coordinates)
    - n_iter: Number of iterations for CRF inference
    
    Returns:
    - Refined segmentation map after CRF
    """
    # Convert image and probabilities to expected data types
    h, w, num_classes = probs.shape
    probs = np.transpose(probs, (2, 0, 1))  # Shape: (num_classes, H, W)

    # Ensure that the probabilities are C-contiguous
    probs = np.ascontiguousarray(probs)

    # Create DenseCRF model
    d = dcrf.DenseCRF2D(w, h, num_classes)
    
    # Unary term (negative log probabilities)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    # Add pairwise Gaussian term (encourages smoothness)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=3)

    # Add pairwise Bilateral term (encourages appearance similarity) using `create_pairwise_bilateral`
    pairwise_bilateral = create_pairwise_bilateral(sdims=sxy_bilateral, schan=srgb_bilateral, img=image, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Perform inference
    Q = d.inference(n_iter)

    # # Convert the CRF output to a segmentation map
    # refined_segmentation = np.argmax(refined_probs, axis=0).reshape(h, w)
    
    
    probs = np.array(Q, dtype=np.float32).reshape((num_classes, h, w))

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