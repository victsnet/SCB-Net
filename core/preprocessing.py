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
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import cv2
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
                 
                if (np.median(x_im) != self.check_all) or (y_im[y_im > 0.0].size > 0.0):
                    X[n] = x_im
                    Y[n] = y_im
                    
                    pairs.append((x_n, y_n))
                    n += 1
                
            else:
                # Extract data and mask patch around point   
                x_im = data[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                y_im = prob_mask[uu][y_n-self.patch_size//2:y_n+self.patch_size//2, x_n-self.patch_size//2:x_n+self.patch_size//2, :]
                 
                if np.median(x_im) != self.check_all or y_im[y_im > 0.0].size > 0:
                    # avoid pairs all == 0.0 or other value
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
                if (np.median(x_im) != self.check_all) or (y_im[y_im > 0.0].size > 0.0):

                    # avoid pairs all == 0.0 or other value
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

def generate_clustered_mask(shape, percentage_ones, cluster_std):
    """
    Generates a binary mask with spatial clusters of 1s.

    Parameters:
    shape (tuple): Shape of the output mask (e.g., (height, width) for a 2D mask).
    percentage_ones (float): Desired percentage of 1s in the mask (value between 0 and 1).
    cluster_std (float): Standard deviation for the Gaussian filter to control cluster size.

    Returns:
    numpy.ndarray: A mask array of 1s and 0s with spatial clusters.
    """
    # Total number of elements in the mask
    total_elements = np.prod(shape)
    
    # Calculate the number of 1s based on the desired percentage
    num_ones = int(total_elements * percentage_ones/100)
    
    # Create an initial random mask with the desired number of 1s
    initial_mask = np.zeros(total_elements, dtype=int)
    initial_mask[:num_ones] = 1
    np.random.shuffle(initial_mask)
    initial_mask = initial_mask.reshape(shape)
    
    # Apply Gaussian filter to introduce clustering
    smoothed_mask = gaussian_filter(initial_mask.astype(float), sigma=cluster_std)
    
    # Normalize and threshold the smoothed mask to get a binary mask
    threshold = np.percentile(smoothed_mask, 100 * (1 - percentage_ones/100))
    clustered_mask = (smoothed_mask > threshold).astype(int)
    
    return clustered_mask

def chessboard_mask(shape, n_pixels):
    mask = np.zeros(shape, dtype=int)
    mask[::2*n_pixels, ::2*n_pixels] = 1
    mask[1::2*n_pixels, 1::2*n_pixels] = 1
    return mask

def filter_points(image, eps=5, min_samples=3):
    """
    Filters out clustered ones in a binary image while maintaining isolated ones.

    Parameters:
    - image: np.array, shape (ny, nx), binary image with zeros and ones
    - eps: float, optional, default=5, maximum distance between two samples for one to be considered as in the neighborhood of the other
    - min_samples: int, optional, default=2, the number of samples (or total weight) in a neighborhood for a point to be considered as a core point

    Returns:
    - filtered_image: np.array, shape (ny, nx), binary image with filtered points
    """

    # check percentage of ones
    if np.median(image) > 0:
        image *= generate_clustered_mask(image.shape, 20, 2.5)
    # Find all coordinates of ones
    points = np.column_stack(np.where(image == 1))

    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Extract the labels
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present
    unique_labels = set(labels)

    # Create an empty list to store filtered points
    filtered_points = []

    # Loop over unique labels
    for label in unique_labels:
        # Get all points for this label
        label_points = points[labels == label]

        if label == -1:
            # Label -1 is for noise points, add them all to filtered_points
            filtered_points.extend(label_points)
        else:
            # For clusters, add only one point (the first one) to filtered_points
            filtered_points.append(label_points[0])

    # Convert filtered points back to numpy array for easier handling
    filtered_points = np.array(filtered_points)

    # Create an empty image to draw filtered points
    filtered_image = np.zeros_like(image)

    # Draw filtered points on the image
    for point in filtered_points:
        filtered_image[point[0], point[1]] = 1

    return filtered_image

def min_dist(idy, idx):

    # Stack the coordinates into a (N, 2) array where each row is a (idy, idx) pair
    coordinates = np.stack((idy, idx), axis=1)

    # Compute all pairwise Euclidean distances between points
    distances = pdist(coordinates)

    return np.min(distances)


def get_indices(prob_mask, dim, title=None, eps=5, min_samples=2, n_iter=10):

    ny, nx = prob_mask.shape[:2]
    pad = np.ones((ny, nx), dtype=float)
    pad[:dim[0]//2, :] = 0.0
    pad[-dim[0]//2:, :] = 0.0
    pad[:, :dim[1]//2] = 0.0
    pad[:, -dim[1]//2:] = 0.0

    max_prob_mask = np.sum(prob_mask, axis=-1)
    cb_mask = chessboard_mask(max_prob_mask.shape, 2)
    max_prob_mask *= cb_mask
    ody, odx = np.where(max_prob_mask > 0.0)
    for i in range(n_iter-1):
        max_prob_mask *= filter_points(max_prob_mask, eps=eps+i, min_samples=min_samples)
    max_prob_mask *= pad
    idy, idx = np.where(max_prob_mask > 0.0)

    # plot figure
    plt.figure(figsize=(10, 10))
    plt.imshow(max_prob_mask, cmap='gray')
    plt.scatter(odx, ody, label=f'samples (count={ody.size})', alpha=0.5, s=10)
    plt.scatter(idx, idy, label=f'centroids (count={idy.size})', alpha=0.9, s=25, marker='x', c='red')
    plt.axis('scaled')
    if title is not None:
        plt.title(title + f' (min_dist={min_dist(idy, idx):.2f})')
    plt.legend()
    plt.xlabel('Units (pixels)')
    plt.ylabel('Units (pixels)')
    plt.grid(linewidth=0.5, alpha=0.7, axis='both', linestyle='--')
    plt.show()

    return idy, idx

class ImageTiling:
    def __init__(self, data, prob_mask_training, prob_mask_validation, dim, min_dist=1, add_padding=True, seed=None, eps=5, min_samples=2, n_train_iter=10, n_val_iter=20):

        if add_padding:
            data = cv2.copyMakeBorder(data, dim[0]//2, dim[1]//2, dim[0]//2, dim[1]//2, cv2.BORDER_CONSTANT, value=0.0)
            prob_mask_training = cv2.copyMakeBorder(prob_mask_training, dim[0]//2, dim[1]//2, dim[0]//2, dim[1]//2, cv2.BORDER_CONSTANT, value=0.0)
            prob_mask_validation = cv2.copyMakeBorder(prob_mask_validation, dim[0]//2, dim[1]//2, dim[0]//2, dim[1]//2, cv2.BORDER_CONSTANT, value=0.0)

        self.data = data
        self.prob_mask_training = prob_mask_training
        self.prob_mask_validation = prob_mask_validation
        self.train_indices = get_indices(prob_mask_training, dim, 'training', eps, min_samples, n_train_iter)
        self.val_indices = get_indices(prob_mask_validation, dim, 'validation', eps, min_samples, n_val_iter)
        self.dim = dim
        self.min_dist = min_dist
        self.seed = seed

    def subset(self, idy, idx, size=None, seed=None, max_iter=30, val_mode=False):

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Set default size if not provided
        if size is None:
            size = idy.size

        if val_mode:
            max_iter = 1

        # Precompute pairwise distances for the entire dataset once
        coords = np.stack((idy, idx), axis=1)
        dist_matrix = squareform(pdist(coords))  # Full pairwise distance matrix


        iter = 0
        while iter < max_iter:
            # Select random subset of indices
            s = np.random.choice(np.arange(idy.size), size=size, replace=False)
            sub_y, sub_x = idy[s], idx[s]
            
            # Extract the submatrix for the selected indices
            sub_dist_matrix = dist_matrix[s][:, s]

            # Check if all pairwise distances in the subset are greater than or equal to min_dist
            if np.all(sub_dist_matrix >= self.min_dist):
                return sub_y, sub_x  # Return valid subset

            iter += 1

        # If no valid subset is found after max_iter, return the last subset
        return sub_y, sub_x

    
    def transformations(self, inputs, k):

        if k == 1:
            out = inputs[::-1, :]  # Vertical flip
        elif k == 2:
            out = inputs[:, ::-1]  # Horizontal flip
        elif k == 3:
            out = np.rot90(inputs)  # Rotate 90 degrees
        elif k == 4:
            out = np.rot90(inputs[::-1, :])  # Rotate 90 degrees and vertical flip
        elif k == 5:
            out = np.rot90(inputs[:, ::-1])  # Rotate 90 degrees and horizontal flip
        elif k == 6:
            out = np.rot90(inputs[::-1, ::-1])  # Rotate 90 degrees and both flips
        else:
            out = inputs

        return out[None]

    def patchify(self, inputs, masks, yy, xx, dim, k=0):
        xpad = dim[1] // 2
        ypad = dim[0] // 2
        usy = usx = 1  # undersampling factor
        ny, nx = masks.shape[:2]
        
        # Correct logical conditions using `and`
        if (k > 8) and (k < 12) and (yy - 2 * ypad > 0) and (yy + 2 * ypad < ny) and (xx - 2 * xpad > 0) and (xx + 2 * xpad < nx):
            xpad *= 2
            ypad *= 2
            usy = usx = 2
            k = k - 8

        elif (k > 12) and (xx - 2 * xpad > 0) and (xx + 2 * xpad < nx):
            xpad *= 2
            ypad *= 1
            usx = 2
            usy = 1
            k = k - 8

        elif (k > 12) and (yy - 2 * ypad > 0) and (yy + 2 * ypad < ny):
            xpad *= 1
            ypad *= 2
            usx = 1
            usy = 2
            k = k - 8

        # Ensure the slicing bounds are within array dimensions
        yy_start = max(0, yy - ypad)
        yy_end = min(ny, yy + ypad)
        xx_start = max(0, xx - xpad)
        xx_end = min(nx, xx + xpad)

        X = inputs[yy_start:yy_end, xx_start:xx_end][::usy, ::usx]
        Y = masks[yy_start:yy_end, xx_start:xx_end][::usy, ::usx]
        
        X = self.transformations(X, k)
        Y = self.transformations(Y, k)
        
        return X, Y

    def aug_type(self, size):
        return [np.random.randint(0, 16) for _ in range(size)]

    def img_mask_pair(self, X, Y, indices, batch_size=None, val_mode=False, seed=None):
        idy, idx = indices
        if batch_size is not None:
            idy, idx = self.subset(idy, idx, batch_size, seed, val_mode=val_mode)
        else:
            batch_size = idy.size
            idy, idx = self.subset(idy, idx, batch_size, seed, val_mode=val_mode)

        ks = self.aug_type(batch_size)
        if val_mode:
            ks = [0] * batch_size            

        Xi = []
        Yi = []
        for yy, xx, k in zip(idy, idx, ks):
            img, msk = self.patchify(X, Y, yy, xx, self.dim, k)
            Xi.append(img)
            Yi.append(msk)

        Xi = np.concatenate(Xi, axis=0)
        Yi = np.concatenate(Yi, axis=0)
        return Xi, Yi

    def training_patches(self, batch_size):
        return self.img_mask_pair(self.data, self.prob_mask_training, self.train_indices, batch_size)

    def validation_patches(self, batch_size, seed=37):
        X_val, Y_val = self.img_mask_pair(self.data, self.prob_mask_validation, self.val_indices, batch_size, val_mode=True, seed=seed)
        _, Y_gt = self.img_mask_pair(self.data, self.prob_mask_training, self.val_indices, batch_size, val_mode=True, seed=seed)
        return X_val, Y_gt, Y_val


def create_distance_grid(probabilities, xi, yi, tile_size, norm=False):
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