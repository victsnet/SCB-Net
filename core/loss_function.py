#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:51:51 2023

@author: silva
"""
import tensorflow as tf
from keras import backend as K
import numpy as np
import tensorflow_addons as tfa


class spatial_losses:

    def __init__(self, dim, ths=0.5, wmatrix=None, proportions=None, hold_out=0.1, block_size=5, fw=[0.5, 0.2, 0.15, 0.15], 
                 fs=[1, 3, 5, 11], fname='dilation', interp_type='classification', declustering=False, declus_kernel_size=3):
        self.dim = dim
        self.ths = ths
        self.wmatrix = tf.convert_to_tensor(wmatrix)
        self.hold_out = hold_out
        self.proportions = tf.cast(tf.convert_to_tensor(proportions), tf.float32)
        self.block_size = 3
        self.interp_type = interp_type
        self.fs = fs
        self.fw = fw
        self.fname = fname
        self.declustering = declustering
        self.declus_kernel_size = declus_kernel_size

    def train_val_dropblock(self, inputs, block_size=3, fill_value=-9999.):
        '''
        DropBlock: A regularization method for convolutional networks
        https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
        '''
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        # Calculate the number of blocks in each dimension
        num_blocks_h = input_shape[1] // block_size
        num_blocks_w = input_shape[2] // block_size

        uniform_dist = tf.random.uniform(
            [input_shape[0], num_blocks_h, num_blocks_w, 1], dtype=inputs.dtype)
        size = tf.shape(inputs)[1:3]  # get spatial dimensions of input tensor
        uniform_dist = tf.image.resize(uniform_dist, size, method='nearest')
        # Create a mask to determine which blocks to drop
        train_mask = uniform_dist > self.drop_samples_rate
        val_mask = uniform_dist <= self.drop_samples_rate

        # create a tensor with the same shape as the image filled with -9999
        fill_matrix = tf.constant(
            fill_value, dtype=inputs.dtype, shape=inputs.shape)
        # use tf.where to replace the pixels below the threshold with -9999
        train_image = tf.where(train_mask, fill_matrix, inputs)
        val_image = tf.where(val_mask, fill_matrix, inputs)

        return (train_image, val_image)
    
    def dropblock(self, inputs, block_size=5, hold_out=0.1):
        '''
        DropBlock: A regularization method for convolutional networks
        https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
        '''
        # Get the shape of the input tensor
        input_shape = tf.shape(inputs)
        # Calculate the number of blocks in each dimension
        num_blocks_h = input_shape[1] // block_size
        num_blocks_w = input_shape[2] // block_size

        uniform_dist = tf.random.uniform(
            [input_shape[0], num_blocks_h, num_blocks_w, input_shape[-1]], dtype=inputs.dtype)
        size = tf.shape(inputs)[1:3]  # get spatial dimensions of input tensor
        uniform_dist = tf.image.resize(uniform_dist, size, method='nearest')
        block_mask = tf.where(uniform_dist > hold_out, 1., 0.)
        output = tf.multiply(inputs, block_mask) 
        
        return output
    
    def declustering_(self, y_true, filter_size=7):

        samples_msk = tf.reduce_max(y_true, axis=-1, keepdims=True)
        avg_msk = tf.nn.avg_pool(samples_msk, ksize=[1, filter_size, filter_size, 1], 
                                       strides=[1, 1, 1, 1], padding='SAME')

        declustering_msk = K.square(samples_msk - avg_msk)
        
        return 1.0 + declustering_msk 
    
    def gabor_filter(self, channels, kernel_size, angle=np.pi/4, sigma=2.0, frequency=0.5, phi=0.0, nmax=1.0, nmin=0.):
        # Create a meshgrid for x and y coordinates
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        
        # Rotate the meshgrid by the specified angle
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        
        # Calculate the Gabor filter using the formula
        gabor_real = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2)) * np.cos(2 * np.pi * frequency * x_rot + phi)
        
        # Convert the real and imaginary parts to TensorFlow tensors
        gabor_real = tf.constant(gabor_real, dtype=tf.float32)
        gabor_real = tf.tile(gabor_real[..., tf.newaxis], [1, 1, channels])
        gabor_real = (gabor_real - tf.reduce_min(gabor_real)) * (nmax - nmin) / (tf.reduce_max(gabor_real) - tf.reduce_min(gabor_real)) + nmin
        
        return gabor_real

    def gauss_kernel(self, channels, kernel_size, sigma=13., nmin=0.5, nmax=1.0,):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        kernel = (kernel - tf.reduce_min(kernel)) * (nmax - nmin) / (tf.reduce_max(kernel) - tf.reduce_min(kernel)) + nmin
    
        return kernel
        
    def dilation2d(self, inputs, kernel_size, sigma=15., gaussian_kernel=False, gabor_kernel=False):
            
        inputs = tf.convert_to_tensor(inputs)
        if gaussian_kernel:

            kernel = self.gauss_kernel(tf.shape(inputs)[-1], kernel_size, sigma)
            output = tf.nn.dilation2d(inputs, filters=kernel, strides=(1, 1, 1, 1), data_format='NHWC', dilations=(1, 1, 1, 1), padding="SAME")
            output = output - tf.ones_like(output)
            
        elif gabor_kernel:
            kernel = self.gabor_filter(tf.shape(inputs)[-1], kernel_size, sigma)
            output = tf.nn.dilation2d(inputs, filters=kernel, strides=(1, 1, 1, 1), data_format='NHWC', dilations=(1, 1, 1, 1), padding="SAME")
            output = output - tf.ones_like(output)
            
        else:
            kernel = tf.ones((kernel_size, kernel_size, inputs.shape[-1]), dtype=inputs.dtype)
            output = tf.nn.dilation2d(inputs, filters=kernel, strides=(1, 1, 1, 1), data_format='NHWC', dilations=(1, 1, 1, 1), padding="SAME")
            output = output - tf.ones_like(output)
            
        return output
        
    def gaussian_blur(self, img, kernel_size=11, sigma=1.75, nmin=0.0, nmax=1.0):
        gaussian_kernel = self.gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
        gaussian_kernel = gaussian_kernel[..., tf.newaxis]
        output = tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                          padding='SAME', data_format='NHWC')
        
        output = (output - tf.reduce_min(output)) * (nmax - nmin) / (tf.reduce_max(output) - tf.reduce_min(output)) + nmin
        
        return 
    
    def mse(self, y_true, y_pred):

        # distance between the predictions and simulation probabilities
        squared_diff = K.square(y_true-y_pred)

        # calculate the loss only where there are samples
        mask = tf.where(y_true > -9999., 1.0, 0.)
        denominator = K.sum(mask)

        # sum of squared differences at sampled locations
        summ = K.sum(squared_diff*mask)
        mse = summ/denominator

        return mse

    def accuracy(self, y_true, y_pred, threshold=0.5):

        # Create a binary mask for pixels above the threshold
        mask = tf.where(y_true > threshold, 1., 0.)

        y_pred = tf.where(y_pred > threshold, 1., 0.)
        y_true = tf.where(y_true > threshold, 1., 0.)
        
        bool_mask = tf.cast(tf.equal(y_true, y_pred), tf.float32) * mask
        masked_acc = tf.reduce_sum(bool_mask, axis=(0, 1, 2), keepdims=True) / (tf.reduce_sum(mask, axis=(0, 1, 2), keepdims=True) + 1e-7)
        weighted_acc = tf.reduce_sum(masked_acc * self.wmatrix)/(tf.reduce_sum(self.wmatrix) + 1e-7)
        

        return weighted_acc   
    
    
    def masked_ssim(self, y_true, y_pred, max_val=1.0, filter_size=11, filter_sigma=1.5, K1=0.01, K2=0.03, eps=1e-6):
        """
        y_true: ground truth
        y_pred: prediction
        filter_sigma: standard deviation of gaussian filter
        filter_size: size of gaussian filter
        max_val: maximum value of input
        K1: constant for stability
        K2: constant for stability
        """
        
        # index_map
        index_map = tf.where(y_true > 0.0, 1.0, 0.0)
        
        # gaussian filter
        kernel = self.gauss_kernel(tf.shape(y_true)[-1], filter_size, filter_sigma)
        kernel = kernel[..., tf.newaxis]
        
        # mask images
        y_true *= index_map
        y_pred *= index_map
        
        # compute mean
        mu_true = tf.nn.depthwise_conv2d(y_true, kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') 
        mu_pred = tf.nn.depthwise_conv2d(y_pred, kernel, strides=[1,1,1,1], padding='SAME',  data_format='NHWC')
        # compute variance
        var_true = tf.nn.depthwise_conv2d(tf.square(y_true), kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - tf.square(mu_true)
        var_pred = tf.nn.depthwise_conv2d(tf.square(y_pred), kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - tf.square(mu_pred)
        # compute covariance
        covar_true_pred = tf.nn.depthwise_conv2d(y_true*y_pred, kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - mu_true*mu_pred
        
        # Set constants for stability
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2
            
        # compute SSIM
        ssim = (2*mu_true*mu_pred + C1)*(2*covar_true_pred + C2)
        denom = (tf.square(mu_true) + tf.square(mu_pred) + C1)*(var_true + var_pred + C2)
        ssim /= (denom + eps)
        # compute mean SSIM
        ssim = tf.reduce_sum(ssim * index_map, keepdims=True) / (tf.reduce_sum(index_map, keepdims=True) + eps)
        # compute loss
        # Avoid NaNs due to unstable divisions
        ssim = tf.clip_by_value(ssim, 0.0, 1.0)
        return ssim
    
    
    def ssim_loss(self, y_true, y_pred, index_map, max_val=1.0, filter_size=11, filter_sigma=3.5, K1=0.01, K2=0.03, eps=1e-6):
        """
        y_true: ground truth
        y_pred: prediction
        filter_sigma: standard deviation of gaussian filter
        filter_size: size of gaussian filter
        max_val: maximum value of input
        K1: constant for stability
        K2: constant for stability
        """
        # gaussian filter
        kernel = self.gauss_kernel(tf.shape(y_true)[-1], filter_size, filter_sigma)
        kernel = kernel[..., tf.newaxis]
        
        # mask images
        y_true *= index_map
        y_pred *= index_map
        
        # compute mean
        mu_true = tf.nn.depthwise_conv2d(y_true, kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') 
        mu_pred = tf.nn.depthwise_conv2d(y_pred, kernel, strides=[1,1,1,1], padding='SAME',  data_format='NHWC')
        # compute variance
        var_true = tf.nn.depthwise_conv2d(tf.square(y_true), kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - tf.square(mu_true)
        var_pred = tf.nn.depthwise_conv2d(tf.square(y_pred), kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - tf.square(mu_pred)
        # compute covariance
        covar_true_pred = tf.nn.depthwise_conv2d(y_true*y_pred, kernel, strides=[1,1,1,1], padding='SAME', data_format='NHWC') - mu_true*mu_pred
        
        # Set constants for stability
        C1 = (K1 * max_val) ** 2
        C2 = (K2 * max_val) ** 2
            
        # compute SSIM
        ssim = (2*mu_true*mu_pred + C1)*(2*covar_true_pred + C2)
        denom = (tf.square(mu_true) + tf.square(mu_pred) + C1)*(var_true + var_pred + C2)
        ssim /= (denom + eps)
        # compute mean SSIM
        ssim = tf.reduce_sum(ssim * index_map, axis=(1, 2), keepdims=True) / (tf.reduce_sum(index_map, axis=(1, 2), keepdims=True) + eps)
        # compute loss
        # Avoid NaNs due to unstable divisions
        loss = 1 - tf.clip_by_value(ssim, 0.0, 1.0)
        return loss
            
    
    def kernel_weights(self, y_true, mask, filter_size=3, eps=1e-7):

        avg_ = tf.nn.avg_pool(y_true, ksize=[1, filter_size, filter_size, 1], 
                                       strides=[1, 1, 1, 1], padding='SAME')
        masked_var = K.square(y_true - avg_) * mask
        class_variance = tf.reduce_sum(masked_var, axis=(0, 1, 2), keepdims=True)/(tf.reduce_sum(mask, axis=(0, 1, 2), keepdims=True) + eps)
        
        return 1.0/(tf.math.log(class_variance + 1.0) + eps)  
    
    def ms_cce_loss(self, y_true, y_pred, mask, gamma=2.0, alpha=1.5, epsilon=1e-7):
        
        # Apply the mask to the images
        #y_true = tf.where(y_true > 0.5, 1.0, 0.0)  
        y_true = tf.multiply(y_true, mask)       
        
        # Define the paramfiltered_y_trueeters for MS-CCE
        wmatrix = 1.
        # gives different weights by class
        if self.wmatrix is not None:
            wmatrix = self.wmatrix
                
        # Compute MS-CCE for each scale
        cce_list = []
        
        # Avoid division by zero
        eps = 1e-9
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        # Compute focal weights
        focal_weights = alpha * tf.pow(1 - y_pred, gamma)
        filtered_y_pred = y_pred
        
        for filter_weight, filter_size in zip(self.fw, self.fs):

            if filter_size > 1:                
                # filter the images for the next scale
                if self.fname == 'dilation':
                    filtered_y_pred = self.dilation2d(y_pred, filter_size)

                elif self.fname == 'gaussian':
                    filtered_y_pred = self.gaussian_blur(y_pred, filter_size, sigma=1.5)
                    
                elif self.fname == 'median':
                    filtered_y_pred = tfa.image.median_filter2d(y_pred, filter_shape=(filter_size, filter_size))
                    
                else:
                    filtered_y_pred = tf.nn.avg_pool(y_pred, ksize=[1, filter_size, filter_size, 1], 
                                                    strides=[1, 1, 1, 1], padding='SAME')

            # weights
            cce = focal_weights * y_true * tf.math.log(filtered_y_pred + epsilon)
            cce *= wmatrix
            cce_sum = -tf.reduce_sum(cce, axis=-1)
            cce_mean = tf.reduce_sum(cce_sum)/tf.reduce_sum(mask)
            cce_list.append(cce_mean * filter_weight)
            
        # final loss
        fcce = tf.reduce_sum(cce_list) 
        
        return fcce

    def spatial_loss(self, y_true, y_pred, kernel=None, training_set=True):
        
        if training_set is False:
            self.hold_out = 0.0
        
        if self.interp_type == 'regression':
            # calculate the loss only where there are samples
            mask = tf.where(y_true > self.ths, 1.0, 0.)
            if self.hold_out > 0.0:
                mask = self.dropblock(mask, self.block_size, self.hold_out)
            loss = self.ms_reg_loss(y_true, y_pred, mask)

        elif self.interp_type == 'classification':
            # calculate the loss only where there are samples
            mask = tf.where(y_true > self.ths, 1.0, 0.)
              
            if self.hold_out > 0.0:
                mask = self.dropblock(mask, self.block_size, self.hold_out)
            loss = self.ms_cce_loss(y_true, y_pred, mask)

        else:
            raise ValueError('Wrong interp_type!')

        return loss




