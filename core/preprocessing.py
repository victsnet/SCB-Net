#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:30:29 2023

@author: silva
"""
import pyproj
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from PIL import Image

def rotate_multichannel_array(image_array, mask_array, max_rotation_degrees=10):
    # Generate a random rotation angle within the specified range
    rotation_angle = np.random.uniform(-max_rotation_degrees, max_rotation_degrees)
    
    # Rotate each channel of the image and mask separately
    rotated_image_array = np.stack([np.array(Image.fromarray(channel).rotate(rotation_angle, Image.BILINEAR)) for channel in image_array.transpose(2, 0, 1)], axis=-1)
    rotated_mask_array = np.stack([np.array(Image.fromarray(channel).rotate(rotation_angle, Image.NEAREST)) for channel in mask_array.transpose(2, 0, 1)], axis=-1)

    return rotated_image_array, rotated_mask_array


class StochasticDownsampling(tf.keras.layers.Layer):
    def __init__(self, pool_size, stride, padding='same', alpha=7):
        super(StochasticDownsampling, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.alpha = alpha
        
    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        self.bsp = tf.shape(input_tensor)[0] 
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        noise_w = tf.random.uniform([1, self.bsp, 1, w], maxval=self.alpha, dtype=tf.float32)
        noise_h = tf.random.uniform([1, self.bsp, h, 1], maxval=self.alpha, dtype=tf.float32)
        noise_mat = tf.transpose(tf.matmul(noise_h, noise_w), (1, 2, 3, 0))
        noise_mat = noise_mat - tf.reduce_max(noise_mat)
        noise_mat = tf.exp(noise_mat)
        masked = tf.multiply(self.input_tensor, noise_mat)
        noise_pool = AveragePooling2D(self.pool_size,
                                                     self.stride,
                                                     self.padding)(noise_mat)
        masked_pool = AveragePooling2D(self.pool_size,
                                                      self.stride,
                                                      self.padding)(masked)
        return masked_pool / noise_pool

def s3pool(input_tensor, pool_size=(2, 2), stride=2, padding='same', alpha=7):
    input_tensor = np.expand_dims(input_tensor, 0)
    l1 = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding=padding)(input_tensor)
    layer = StochasticDownsampling(pool_size, stride, padding, alpha)
    return layer(l1).numpy()[0]


def Undersampling(image, mask, undersample_by, avg_pool=False, stochastic=True):
    
    resampled_image, resampled_mask = image, mask
    if undersample_by > 1:
        
        if avg_pool:
            resampled_image = MaxPooling2D(pool_size=(undersample_by, undersample_by))(np.expand_dims(image, 0)).numpy()[0]
            resampled_mask = MaxPooling2D(pool_size=(undersample_by, undersample_by))(np.expand_dims(mask, 0)).numpy()[0]
        else:
            yy = np.arange(0, image.shape[0], undersample_by)
            xx = np.arange(0, image.shape[1], undersample_by)
            
            idx, idy = np.meshgrid(xx, yy)
            
            ny = idy.shape[0]
            nx = idy.shape[1]
            
            if stochastic:
                resampled_image =  s3pool(image, alpha=7.0)
            else:
                resampled_image = image[idy.ravel(), idx.ravel(), :].reshape((ny, nx, image.shape[-1]))
            resampled_mask = mask[idy.ravel(), idx.ravel(), :].reshape((ny, nx, mask.shape[-1]))

    return resampled_image, resampled_mask

def normalize_coordinates(x, y, z=0):
    xn = (x - x.min())/(x.max()-x.min())
    yn = (y - y.min())/(y.max()-y.min())
    
    zn = 0
    if z.any() > 0:
        zn = (z - z.min())/(z.max()-z.min())
    
    return (xn, yn, zn)

def latlong_to_cartesian(lat, long):
    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    long_rad = np.radians(long)

    # Equatorial radius of the Earth in meters
    equatorial_radius = 6_378_137

    # Calculate the x, y, and z coordinates
    x = equatorial_radius * np.cos(lat_rad) * np.cos(long_rad)
    y = equatorial_radius * np.cos(lat_rad) * np.sin(long_rad)
    z = equatorial_radius * np.sin(lat_rad)        

    # Return a tuple of the Cartesian coordinates
    return np.array([x, y, z])

def geodetic_to_cartesian(lat, lon, elevation):
    # WGS 84 reference ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening

    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat_rad) ** 2)

    # Calculate Cartesian coordinates
    x = (N + elevation) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + elevation) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - f) ** 2 + elevation) * np.sin(lat_rad)

    return np.array([x, y, z])


#%% transform reference system and normalize x,y
def transform_norm_crs(x, y, source_crs, target_crs):
    latlon2utm = pyproj.Transformer.from_crs(source_crs, target_crs)
    x_utm, y_utm = latlon2utm.transform(x, y)
    
    xn, yn = normalize_coordinates(x_utm, y_utm)
    
    return (xn, yn)

#%% preprocess gridded-data

def standardize_grid(grid, dtype=np.float32, mask=None):
    
    dim = grid.shape
    loc = np.argmin(dim)
    
    data = np.zeros(dim)
    for n in range(grid.shape[loc]):
        
        if loc == 0:
            data[n] = (grid[n]-np.nanmean(grid[n]))/np.nanstd(grid[n])
            
        if loc == 2:
            if mask is None:
                data[:, :, n] = (grid[:, :, n]-np.nanmean(grid[:, :, n]))/np.nanstd(grid[:, :, n])
            else: 
                data[:, :, n] = (grid[:, :, n]-np.nanmean(grid[:, :, n][grid[:, :, n] > mask]))/\
                    np.nanstd(grid[:, :, n][grid[:, :, n] > mask])
           
    # defining the dtype
    data = np.array(data, dtype=dtype)
    return data

#%% create mask from dataset

def msk_from_dset(dataset, out=-2, dtype=np.float32):

    prob_mask = np.stack([dataset[var].data for var in dataset.variables][:out])
    prob_mask = np.array(prob_mask, dtype=dtype)
    
    return prob_mask

#%% create image tiles

class image_tiling:
    
    def __init__(self, data, patch_size, n_classes, underscaling=[1, 2, 3], check_all=0.0, stochastic=False, 
                 random_flip=False, random_rotate=False):
        self.data = data
        self.patch_size = patch_size
        self.n_classes = n_classes 
        self.stochastic = stochastic
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.check_all = check_all
        
        underscaling = np.array(underscaling)
        n_elems = underscaling[underscaling > 1].size
        underscaling = np.concatenate((np.ones(n_elems), underscaling))
        self.underscaling = np.int16(underscaling)
        

    def min_dist(self, x_n, y_n, pairs):
        distances = [9999.]
        for pair in pairs:
            x_f, y_f = pair
            distances.append(np.sqrt(np.power(x_n-x_f, 2) + np.power(y_n-y_f, 2)))

        return np.min(distances)


    def training_patches(self, prob_mask, patch_num, overlap=.9, return_argmax=False,
                                 binary=False, add_noise=0.1, max_it=10000):
        
        min_dist_to_sample = int(self.patch_size * (1. - overlap))
        
        self.patch_num = patch_num
        self.prob_mask = prob_mask

        if self.prob_mask.shape[0] == np.min(self.prob_mask.shape):
           self.prob_mask = self.prob_mask.swapaxes(1, 0).swapaxes(1, 2)


        X = np.zeros((self.patch_num, self.patch_size, self.patch_size, self.data.shape[-1]), dtype=np.float32)
        Y = np.zeros((self.patch_num, self.patch_size, self.patch_size, self.n_classes), dtype=np.float32)
        
        data, prob_mask = {}, {}
        mmax = int(np.max(self.underscaling))+1
        for uu in self.underscaling:
            data[uu], prob_mask[uu] = Undersampling(self.data, self.prob_mask, undersample_by=uu, stochastic=self.stochastic)
            if self.random_flip:
                data[uu+mmax], prob_mask[uu+mmax] = Undersampling(self.data[:, ::-1], self.prob_mask[:, ::-1], undersample_by=uu,
                                                              stochastic=self.stochastic)
                
            if uu == 1:
                if self.random_rotate:
                    for angle in range(5):
                        data[f"{uu}_{angle}"], prob_mask[f"{uu}_{angle}"] = rotate_multichannel_array(data[uu], prob_mask[uu])
                        
            
        pairs = []
        n = interations = 0
        while(n < self.patch_num):
            # register interation
            interations += 1
            
            # get dims
            uu = np.random.choice(self.underscaling)
            (t_max, x_max, _) = data[uu].shape
            (t_min, x_min) = (0, 0)
                    
            if self.stochastic:
                data[uu], prob_mask[uu] = Undersampling(self.data, self.prob_mask, undersample_by=uu, stochastic=self.stochastic)
                
            if self.random_flip:
                if n % 2 == 0:
                    uu += mmax
                    
            if self.random_rotate and (uu == 1):
                if n % 2 == 0:
                    uu = f"{uu}_{np.random.randint(5)}"
                    t_min, x_min = self.patch_size//4, self.patch_size//4
                    x_max -= self.patch_size//4
                    t_max -= self.patch_size//4

            # Select random point in data (not too close to edge)   
            x_n = random.randint(self.patch_size//2+x_min, x_max-self.patch_size//2)
            y_n = random.randint(self.patch_size//2+t_min, t_max-self.patch_size//2)
            
            
            if (uu == 1) & (self.min_dist(x_n, y_n, pairs) > min_dist_to_sample):
                # Extract data and mask patch around point   
                x_im = data[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                y_im = prob_mask[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                 
                if np.all(x_im == self.check_all) or np.all(y_im == self.check_all):
                    pass
                else:
                    X[n] = x_im
                    Y[n] = y_im
                    
                    pairs.append((x_n, y_n))
                    n += 1
                
            else:
                # Extract data and mask patch around point   
                x_im = data[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                y_im = prob_mask[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                 
                if np.all(x_im == self.check_all) or np.all(y_im == self.check_all):
                    pass
                else:
                    X[n] = x_im
                    Y[n] = y_im
                    
                    pairs.append((x_n, y_n))
                    n += 1

            if interations >= max_it:
                break
            
        if return_argmax:
            Y = np.argmax(Y,  axis=3)

        if binary:
            Y = np.where(Y >= .5, 1., 0.)

        if add_noise > 0.0:
            pass
        # save pairs list
        self.pairs = pairs

        return X[:(n//2)*2], Y[:(n//2)*2]


    def validation_patches(self, prob_mask, patch_num, overlap=.95, return_argmax=False,
                                   binary=False, add_padding=False, n_classes=None, max_it=10000):
        
        min_dist_to_sample = int(self.patch_size * (1. - overlap))
        self.patch_num = patch_num
        self.prob_mask = prob_mask

        if self.prob_mask.shape[0] == np.min(self.prob_mask.shape):
                    self.prob_mask = self.prob_mask.swapaxes(1, 0).swapaxes(1, 2)
                    
        if n_classes is not None:
            self.n_classes = n_classes

        X = np.zeros((self.patch_num, self.patch_size, self.patch_size, self.data.shape[-1]), dtype=np.float32)
        Y = np.zeros((self.patch_num, self.patch_size, self.patch_size, self.n_classes), dtype=np.float32)

        (t_max, x_max, _) = self.data.shape
        pairs = []

        n = interations = 0
        while(n < self.patch_num):
            # register interation
            interations += 1

            # Select random point in data (not too close to edge)   
            x_n = random.randint(self.patch_size//2, x_max-self.patch_size//2)
            y_n = random.randint(self.patch_size//2, t_max-self.patch_size//2)
            

            if self.min_dist(x_n, y_n, pairs) > min_dist_to_sample:
                # Extract data and mask patch around point   
                x_im = self.data[y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                y_im = self.prob_mask[y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                 
                # avoid pairs all == 0.0 or other value
                if np.all(x_im == self.check_all) or np.all(y_im == self.check_all):
                    pass
                else:
                    X[n] = x_im
                    Y[n] = y_im
                    
                    pairs.append((x_n, y_n))
                    n += 1

            if interations >= max_it:
                break

        if return_argmax:
            Y = np.argmax(Y,  axis=3)

        if binary:
            Y = np.where(Y >= .5, 1., 0.)
  
        # save pairs list
        self.pairs = pairs

        return X[:(n//2)*2], Y[:(n//2)*2]
    
def create_distances_grid(probabilities, xi, yi, tile_size, norm=False):
    grid = np.max(probabilities, axis=-1)
    ny, nx = grid.shape

    pad = 2*tile_size
    output = np.pad(np.zeros((ny, nx)), pad_width=pad, mode='constant', constant_values=0.)
    grid = np.pad(grid, pad_width=pad, mode='constant', constant_values=0.)
    xi = np.pad(xi, pad_width=pad, mode='constant', constant_values=np.mean(xi))
    yi = np.pad(yi, pad_width=pad, mode='constant', constant_values=np.mean(yi))

    ymax, xmax = grid.shape

    for y in tqdm(range(tile_size, ymax-tile_size, tile_size)):
        for x in range(tile_size, xmax-tile_size, tile_size):
            subset = grid[y-tile_size:y+tile_size, x-tile_size:x+tile_size]
     
            if len(subset[subset > 0]) > 0:
                
                ya = yi[y-tile_size//2:y+tile_size//2, x-tile_size//2:x+tile_size//2].ravel() 
                xa = xi[y-tile_size//2:y+tile_size//2, x-tile_size//2:x+tile_size//2].ravel()
                
                # get coordinates from sampled locations
                cond = subset > 0.
                yb = yi[y-tile_size:y+tile_size, x-tile_size:x+tile_size][cond]
                xb = xi[y-tile_size:y+tile_size, x-tile_size:x+tile_size][cond]                
                               
                y_square_diff = np.square(np.subtract.outer(ya, yb))
                x_square_diff = np.square(np.subtract.outer(xa, xb))
                distances = np.sqrt(y_square_diff + x_square_diff)
                distances = np.min(distances, axis=-1).reshape((tile_size, tile_size))
                output[y-tile_size//2:y+tile_size//2, x-tile_size//2:x+tile_size//2] = distances

    output = output[pad:-pad, pad:-pad]
    
    if norm:
        output = (output - np.min(output))/(np.max(output) - np.min(output))
    
    return output