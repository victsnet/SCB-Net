#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:28:22 2023

@author: silva
"""

import numpy as np
from numpy import rot90   
import tensorflow as tf 
import rioxarray
from rioxarray import open_rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import textdistance
from translate import Translator
from tqdm.notebook import tqdm
from  matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex
import pandas as pd

def merge_xarrays(paths, scaling):
    '''
    

    Parameters
    ----------
    paths : LIST
        list of paths.
    scaling : float
        rescaling factor.

    Returns
    -------
    x : array
        x coordinates.
    y : array
        y coordinates.
    data_grid : array
        grid of variables.

    '''
    
    data_grid = []
    for path in paths:
        if int(scaling) == 1:
            ds = rioxarray.open_rasterio(path)
            
        else:
            xds = rioxarray.open_rasterio(path)

            # resample the data
            new_width = int(xds.rio.width * scaling)
            new_height = int(xds.rio.height * scaling)

            ds = xds.rio.reproject(
                xds.rio.crs,
                shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )

        x, y = np.meshgrid(ds.x.data, ds.y.data)
        data = ds.data
        data_grid.append(data)

    # stack arrays
    data_grid = np.vstack(data_grid)
    
    
    return x, y, data_grid   


def normalize(x, lim=255.):
    return (x-np.min(x))/(np.max(x)-np.min(x))*lim

def adjust_rgb(img, perc_init=5, perc_final=95, nchannels=3):
    
    dim = img.shape
    adjusted_img = np.zeros((dim))
    
    if dim[-1] == nchannels:
        
        for n in range(nchannels):
            channel = img[:, :, n]
            perc_i = np.percentile(channel, perc_init)
            perc_f = np.percentile(channel, perc_final)
            channel = np.clip(channel, perc_i, perc_f)
            channel = normalize(channel, 1.)
            adjusted_img[:, :, n] = channel
        
    else:
        raise ValueError(f'The shape should be (M, N, {nchannels}).')
        
        
    return adjusted_img

def hillshade(array, azimuth, angle_altitude):

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) \
             + np.cos(altituderad) * np.cos(slope) \
             * np.cos(azimuthrad - aspect)
    hillshade_array = 255 * (shaded + 1) / 2

    return hillshade_array

def plot_cfm(cfm, labels, cmap='viridis', norm=False, savefig=False, path=''):
    
    fig, ax = plt.subplots(figsize=(len(labels), len(labels)-2))
    
    if norm:
        kron_cfm = np.kron(np.sum(cfm, axis=1), np.ones((cfm.shape[0], 1))).T
        cfm = np.int64(cfm)
        kron_cfm = np.int64(kron_cfm)
        cfm = cfm/kron_cfm
        
        im = ax.matshow(cfm, cmap=cmap)
        for (i, j), z in np.ndenumerate(cfm):
            ax.text(j, i, f'{np.around(z, 2)}', ha='center', va='center')
        fig.colorbar(im, fraction=0.05).set_label(label='Precision', size=len(labels))
        
    else:
        im = ax.matshow(cfm, cmap=cmap)
        for (i, j), z in np.ndenumerate(cfm):
            ax.text(j, i, f'{int(z)}', ha='center', va='center')
        fig.colorbar(im, fraction=0.05).set_label(label='Samples', size=len(labels))
    
    plt.yticks(range(len(labels)), labels, fontsize=11)
    plt.xticks(range(len(labels)), labels, rotation=55, fontsize=11, ha='left')
    plt.xlabel('Predicted label', fontsize=len(labels)+2)
    plt.ylabel('True label', fontsize=len(labels)+2)
    
    plt.gca().set_aspect('equal')
    if savefig:
        fig.savefig(path, dpi=400, bbox_inches='tight')
    plt.show()
    
def reinterp_xarrays(ref_path, paths, scaling=1, method='nearest'):
        
    # open reference dataset
    if int(scaling) == 1:
        with open_rasterio(ref_path) as ref_ds:
            x0, y0 = ref_ds.x.data, ref_ds.y.data
            
    else:
        with open_rasterio(ref_path) as xds:

            # resample the data
            new_width = int(xds.rio.width * scaling)
            new_height = int(xds.rio.height * scaling)
    
            ref_ds = xds.rio.reproject(
                xds.rio.crs,
                shape=(new_height, new_width),
                resampling=Resampling.bilinear,
            )
            x0, y0 = ref_ds.x.data, ref_ds.y.data
                   
    
    grid_bandvalues = []
    for path in paths:
        with open_rasterio(path) as ds:
            ds = ds.interp(x=x0, y=y0, method=method)
            grid_bandvalues.append(ds.data)
            
    grid_bandvalues = np.vstack(grid_bandvalues)
    
    # ref mesh
    x0, y0 = np.meshgrid(ref_ds.x.data, ref_ds.y.data)
    
    return x0, y0, grid_bandvalues

def create_wmask(path, scaling, median_flt=False):

    # import multispec data
    xds = rioxarray.open_rasterio(path)

    # resample the data
    new_width = int(xds.rio.width * scaling)
    new_height = int(xds.rio.height * scaling)

    xds = xds.rio.reproject(
        xds.rio.crs,
        shape=(new_height, new_width),
        resampling=Resampling.bilinear,
    )
    
    s2 = xds.data
    ## calculate NDWI
    ndwi = (s2[1]-s2[3]+0.1)/(s2[1]+s2[3]+0.1)

    # clustering
    scaler = StandardScaler()
    XX = np.stack((ndwi, ndwi))
    nz, ny, nx = XX.shape
    XX = np.vstack(XX.reshape((nz, ny*nx))).swapaxes(0, 1)
    XX = scaler.fit_transform(XX)
    kmeans = KMeans(n_clusters=2, n_init='auto').fit(XX)
    clusters = kmeans.labels_.reshape((ny, nx))

    # mask creation
    idy1, idx1 = np.where(clusters == 0)
    idy2, idx2 = np.where(clusters == 1)
    cid = np.argmax([np.mean(ndwi[idy1, idx1]), np.mean(ndwi[idy2, idx2])])
    mask = np.where(clusters == cid, 1., 0.)
    if median_filter:
        mask = median_filter(mask, size=3)
    
    return mask


def calc_text_similarity(string1, string2, important_word_length=10):
    
    list_words = string2.split(' ')
    word_sim = 0.
    
    if len(list_words) > 1:
        important_word = list_words[0]
        if len(important_word) >= 9:
            important_word = list_words[1]
            
        jaro_sim = textdistance.jaro_winkler(string1, important_word)
        jaccard_sim = textdistance.jaccard(string1, important_word)
        ratcliff_sim = textdistance.ratcliff_obershelp(string1, important_word)
        word_sim = jaro_sim*jaccard_sim*ratcliff_sim
        
    jaro_sim = textdistance.jaro_winkler(string1, string2)
    jaccard_sim = textdistance.jaccard(string1, string2)
    ratcliff_sim = textdistance.ratcliff_obershelp(string1, string2)
    
    return (jaro_sim*jaccard_sim*ratcliff_sim)+word_sim


def compare2catalog(original_string, catalog, translate=False, from_lang='fr', to_lang='en'):
    
    translator = Translator(from_lang=from_lang, to_lang=to_lang)
    # translate
    string2 = translator.translate(original_string)
    string2 = string2.lower()
    
    similarities = []
    for string1 in catalog.text:

        string1 = string1.lower()
        similarities.append(calc_text_similarity(string1, string2, len(original_string.split(' ')[0])))

    # calc similarity
    argmx = np.argmax(similarities)
    max_sim = np.max(similarities)
        
    return (argmx, max_sim, string2)

def cmap_from_labels(labels_list, path_catalog, sep=',', translate=False, from_lang='fr', to_lang='en'):
    
    catalog = pd.read_csv(path_catalog, sep=sep)
    hex_codes = catalog.apply(lambda x: to_hex(np.asarray([x['r'], x['g'], x['b'], 255])/255.0), axis=1)
    
    hex_list = []
    for label in labels_list:
        argmx, _, _ = compare2catalog(label, catalog, translate=translate, from_lang=from_lang, to_lang=to_lang)
        hexc = hex_codes[argmx]
        
        if hexc in hex_list:
            modcolor = np.min(([255., 255., 255.], catalog[['r', 'g', 'b']].values[argmx]+np.array([18., 30., 19.])), axis=0)
            new_rgb = np.concatenate((modcolor, np.array([255.])))/255.
            new_hex = to_hex(new_rgb)
            hex_list.append(new_hex)
        else:
            hex_list.append(hexc)
        
    return ListedColormap(hex_list)


def export_tif(ds, crs, path, x='lon', y='lat'):

    import rasterio
    import warnings
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    
    ds.rio.set_crs(crs, inplace=True)
    ds.rio.to_raster(path, driver='GTiff')
    
    x = ds[x].data
    y = ds[y].data
    src = rasterio.open(path)
    # get meta dict
    meta = src.meta

    # calculate pixel size
    pixel_width, pixel_heigth = np.abs(np.mean((x[:, 0]-x[:, 1]))), np.abs(np.mean((y[0]-y[1])))


    # add new affine
    meta['transform'] = rasterio.Affine(float(pixel_width), float(0.0), x.min(),
                    0.0, float(-pixel_heigth), float(y.max()))

    # export to tiff
    with rasterio.open(path, 'w', **meta) as raster:
        # If array is in  (y, x, z) order (cols, rows, bands)
        source = src.read()
        raster.write(source)
        
class data_augmentation:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        

    def rotation(self, nrot=[0, 1, 2, 3], perc=1.):

        Xaug = []
        Yaug = []
        
        for n in nrot:
            Xaug.append(rot90(self.X, n, axes=(1, 2)))
            Yaug.append(rot90(self.Y, n, axes=(1, 2)))            
        
        n_generated_samples = int(self.X.shape[0] + perc*self.X.shape[0])
        Xaug = np.concatenate(Xaug)[:n_generated_samples]
        Yaug = np.concatenate(Yaug)[:n_generated_samples]
        size = Xaug.shape[0]
        
        shuffle = np.random.choice(np.arange(0, size, 1, dtype=np.int16), size=size, replace=False)                
                
        self.X = Xaug[shuffle]
        self.Y = Yaug[shuffle]

        return self.X, self.Y    
    
    def noise(self, var=0.05):

        self.X = self.X + np.random.normal(np.nanmean(self.X), var, size=self.X.shape)

        return self.X  
    
def find_labels(original_labels, litho_dict, mmax=3, fill_value='XXXX'):
    
    from numpy import array, argmax, full
    
    
    size = original_labels.size
    priority = full((size, mmax), fill_value, dtype=object) 
    
    for n in tqdm(range(size)):
        # possible labels
        plabels = []
        # labels' length
        length = []
        # indexes
        indexes = []

        for label in litho_dict.CODE.values:
            idx = original_labels[n].find(label)
            if idx in range(6):
                label_ = original_labels[n][idx]
                len_ = len(label)
                plabels.append(label)
                length.append(len_)
                indexes.append(idx)
        
        
        if len(indexes) > 0:
            # use length and order to define the priority to the words in list
            args_idx = argmax(array(length) + (100 - array(indexes)))

            # tranform list to array
            plabels = array(plabels)[args_idx]
            list_size = len(plabels)


            if list_size == 1:

                plabels = array([plabels[0], fill_value, fill_value])

            if list_size == 2:

                plabels = array([plabels[0], plabels[:2], fill_value])

            if list_size >= 3:  
                
                plabels = array([plabels[0], plabels[:2], plabels[:3]])
                

            priority[n] = plabels
        
        
    return priority

def rem_startwith(strings_list, join_with='-'):
    
    for string_1 in strings_list:
        for string_2 in strings_list:
            # check if words of different length start with the same letters
            if (len(string_1) == 3) & (len(string_2) == 2):
                    if string_1[:2] == string_2[:2]:
                        strings_list.remove(string_2)
                        
    strings_list = join_with.join(strings_list)
                      
    return strings_list 

def get_labels(original_labels, litho_dict, lgmin=2, fill_empty_with='XXXX'):
    
    labels_list = []
    for orig_label in original_labels:
        store_labels = []
        for label in litho_dict.CODE.values:
            if (len(label) >= lgmin) & (label in orig_label):
                store_labels.append(label) 
        labels_list.append(rem_startwith(store_labels)) 

    labels_list = np.where(labels_list == '', fill_empty_with, labels_list)

    return labels_list 

    
def get_samples_mask(xs, ys, x, y, pixel_size, buffer=2.5):
    
    from numpy import full, sqrt, square, where

    n_samples = xs.shape[0]
    ny, nx = x.shape


    mask = np.zeros((ny, nx))
    xp = full(x.shape, xs[0])
    yp = full(y.shape, ys[0])
    
    for n in tqdm(range(n_samples)):
        
        xp[:] = xs[n]
        yp[:] = ys[n]
        distances = sqrt(square(x-xp)+square(y-yp))
        mask[distances < buffer * pixel_size] = 1.
        
    return mask


def calc_local_probs(df, sublabels, X, Y, pixel_size, column='code_r3'):

    n_classes = sublabels.size
    ssize = X.ravel().size
    output = np.full((n_classes, ssize), -9999, dtype=np.float32)

    for s in tqdm(range(ssize)):

        x = X.ravel()[s]
        y = Y.ravel()[s]
        radius = pixel_size*(np.sqrt(2.0)/2.0)
        bbox = Point([(x, y)]).buffer(radius)
        # condition
        cond = np.where((df.lon > (x-radius)) & (df.lon < (x+radius)) &\
            (df.lat > (y-radius)) & (df.lon < (y+radius)))[0]

        # create subset
        subset = df.iloc[cond].clip(bbox)
    
        # check if subset is empty
        if subset.shape[0] > 0:
            # calculate probabilities
            prob_array = np.zeros((n_classes,), dtype=np.float32)
            labels, counts = np.unique(subset[column], return_counts=True)
            local_prob = counts/np.sum(counts) # this will calculate the local probability for each class
                
            for lb, prob in zip(labels, local_prob):
                if lb in sublabels:
                    loc = np.where(sublabels == lb)[0][0]
                    prob_array[loc] = prob 
                    
            # add probabilities to output array
            output[:, s] = prob_array

    # reshape output 
    output = output.reshape(n_classes, *X.shape)
    return output

class pixel_probs:
    
    def __init__(self, df, sublabels, X, Y, mask, pixel_size, column='code_r2', parallel=False):
        
        n_classes = sublabels.size
        ssize = X.ravel().size
        output = np.full((n_classes, ssize), -9999., dtype=np.float32)
        weights = np.full((1, ssize), 0., dtype=np.float32)
        self.df = df
        self.sublabels = sublabels
        self.X = X
        self.Y = Y
        self.mask = mask.ravel()
        self.pixel_size = pixel_size
        self.column = column
        self.output = output
        self.weights = weights
        self.n_classes = n_classes
        self.parellel = parallel
        self.process_number = 0
        self.s = 0

        
    def local_probs(self, position=0):

        if self.mask[position] > 0:
            
            x = self.X.ravel()[position]
            y = self.Y.ravel()[position]
            bf = self.pixel_size*(np.sqrt(2.0)/2.0)
            bbox = Point([(x, y)]).buffer(bf)
            # condition
            cond = np.where((self.df.lon > (x-bf)) & (self.df.lon < (x+bf)) &\
                    (self.df.lat > (y-bf)) & (self.df.lon < (y+bf)))[0]

            # create subset
            subset = self.df.iloc[cond].clip(bbox)

            # check if subset is empty
            if subset.shape[0] > 0:

                # calculate probabilities
                prob_array = np.zeros((self.n_classes,), dtype=np.float32)
                labels, counts = np.unique(subset[self.column], return_counts=True)
                local_prob = counts/np.sum(counts) # this will calculate the local probability for each class
                weight = 1./np.sum(counts) # pixel weight for declustering

                for lb, prob in zip(labels, local_prob):
                    if lb in self.sublabels:
                        loc = np.where(self.sublabels == lb)[0][0]
                        prob_array[loc] = prob

                # add probabilities to output array
                self.output[:, position] = prob_array
            
    def create_batches(self, n_processes=50):
        
        batches = []
        ssize = self.X.ravel().size
        intervals = np.linspace(0, ssize, n_processes+1, dtype=int)
        for i in range(n_processes):
            batches.append(np.arange(intervals[i], intervals[i+1], 1, dtype=int))
            
        self.batches = batches
        print(len(self.batches))
            
    def parellel_processing(self, process_number, return_prob_dict, return_weight_dict, declustering=False, printit=False):
        
        self.parellel = True
        process = self.batches[process_number]
        if printit:
            print(f'Working on process {process_number+1}')
        self.process_number += 1
        for s in tqdm(process):
            self.s = s
            self.local_probs(position=s)
            
        return_prob_dict[process_number] = self.output
        if declustering:
            return_weight_dict[process_number] = self.weights
        
    def get_output(self, reshape=True):
        out = self.output
        if reshape:
            out = out.reshape(self.n_classes, *self.X.shape)
            
        return out
    

def min_dist(x_n, y_n, pairs):
    distances = []
    for pair in pairs:
        x_f, y_f = pair
        distances.append(np.sqrt(np.power(x_n-x_f, 2) + np.power(y_n-y_f, 2)))
        
    return np.min(distances)
    
def spatial_blocks_split(df, columm, index, sublabels, radius, xl='lon', yl='lat', n_clusters=5, min_dist_between_centers=5000, meter2degrees=True):
    
    import warnings
    warnings.filterwarnings("ignore")
    
    df = df.loc[index].copy()
    df.reset_index(drop=True, inplace=True)
    
    if meter2degrees:
        radius /= 111139
        min_dist_between_centers /= 111139

    # rows idxs of test set
    test_idx = []
    pairs = []

    n = iterations = 0
    while(len(test_idx) < n_clusters):
        
        iterations += 1
        idx = np.random.choice(df.index.values, size=1, replace=False)
        x_n, y_n = (df[xl].values[idx], df[yl].values[idx])
        
        if n == 0:
            pairs.append((x_n, y_n))
            test_idx.append(idx)
            n += 1
    
        if(n >= 1) and (min_dist(x_n, y_n, pairs) >= min_dist_between_centers):
            pairs.append((x_n, y_n))
            test_idx.append(idx)
            n += 1
            
        if iterations > 10000:
            break
                   

    test_idx = np.concatenate(test_idx)

    test_df = df.loc[test_idx].copy()
    test_df.reset_index(drop=True, inplace=True)
    
    # set buffer mask
    mask = test_df.buffer(radius)
    
    # validation indexes        
    val_idx = df.clip(mask).index
    # set train idx
    train_idx = np.setdiff1d(df.index.values, val_idx)
    train_df = df.loc[train_idx].copy()
    train_df.reset_index(drop=True, inplace=True)

    # create validation set
    val_df = df.loc[val_idx].copy()
    val_df.reset_index(drop=True, inplace=True)
        
    return train_df, val_df, train_idx, val_idx

def spatialblocks(inputs, block_size=5):
    '''
    DropBlock: A regularization method for convolutional networks
    https://proceedings.neurips.cc/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
    '''
    # Get the shape of the input tensor
    input_shape = tf.shape(inputs)
    size = input_shape[1:3]
    # Calculate the number of blocks in each dimension
    num_blocks_h = input_shape[1] // block_size
    num_blocks_w = input_shape[2] // block_size
    # Create a mask to determine which blocks to drop
    uniform_dist = tf.random.uniform([1, num_blocks_h, num_blocks_w, 1], dtype=inputs.dtype)
    uniform_dist = tf.image.resize(uniform_dist, size, method='nearest')
    
    return uniform_dist.numpy()[:, :, :, 0]