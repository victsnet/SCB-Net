#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:28:22 2023

@author: silva
"""

import numpy as np
from numpy import rot90    
import rioxarray
from rioxarray import open_rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import textdistance
from translate import Translator
from  matplotlib.colors import ListedColormap
from matplotlib.colors import to_hex
import tensorflow as tf
import pandas as pd

def spatialblock_split(inputs, block_size=5, hold_out=0.1):
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
    train_mask = tf.where(uniform_dist > hold_out, 1., 0.)
    val_mask = tf.where(uniform_dist <= hold_out, 1., 0.)
    
    # outputs
    train_output = tf.multiply(inputs, train_mask).numpy() 
    val_output = tf.multiply(inputs, val_mask).numpy() 
    
    return (train_output, val_output)

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

    '''
    Parameters
    ----------
    img : array
        image array.
    perc_init : int, optional
    initial percentile. The default is 5.    
    perc_final : int, optional
    final percentile. The default is 95.
    nchannels : int, optional
    number of channels. The default is 3.
    '''
    
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

def plot_cfm(cfm, labels, cmap='viridis', norm=False, savefig=False, figsize=None, path=''):
    if figsize is None:
        figsize = (len(labels)-4, len(labels)-6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if norm:
        kron_cfm = np.kron(np.sum(cfm, axis=1), np.ones((cfm.shape[0], 1))).T
        cfm = np.int64(cfm)
        kron_cfm = np.int64(kron_cfm)
        cfm = cfm/(kron_cfm+1e-8)
        
        im = ax.matshow(cfm, cmap=cmap, vmin=0.05, vmax=0.95)
        for (i, j), z in np.ndenumerate(cfm):
            ax.text(j, i, f'{np.around(z, 2)}', ha='center', va='center')
        fig.colorbar(im, fraction=0.05).set_label(label=u'Normalized values', size=12)
        
    else:
        im = ax.matshow(cfm, cmap=cmap)
        for (i, j), z in np.ndenumerate(cfm):
            ax.text(j, i, f'{int(z)}', ha='center', va='center')
        fig.colorbar(im, fraction=0.05).set_label(label='Samples', size=12)
    
    plt.yticks(range(len(labels)), labels, fontsize=13)
    plt.xticks(range(len(labels)), labels, rotation=55, fontsize=13, ha='left')
    plt.xlabel('Predicted lithology', fontsize=13)
    plt.ylabel('True lithology', fontsize=13)
    
    plt.gca().set_aspect('equal')
    if savefig:
        fig.savefig(path, dpi=350, bbox_inches='tight')
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
                   
    channels = []
    for path in paths:
        with open_rasterio(path) as ds:
            ds = ds.interp(x=x0, y=y0, method=method)
            channels.append(ds.data)
    channels = np.vstack(channels)
    
    # ref mesh
    x0, y0 = np.meshgrid(ref_ds.x.data, ref_ds.y.data)
    
    return x0, y0, channels

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
        

    def rotation(self, nrot=[1, 2, 3], perc=1.):

        Xaug = []
        Yaug = []
        
        for n in nrot:
            Xaug.append(rot90(self.X, n, axes=(1, 2)))
            Yaug.append(rot90(self.Y, n, axes=(1, 2)))            
        
        size = self.X.shape[0]
        n_generated_samples = int(perc*size)
        
        Xaug = np.concatenate(Xaug)
        Yaug = np.concatenate(Yaug)
        shuffle = np.random.choice(np.arange(0, size, 1, dtype=np.int16), size=size, replace=False)   
        self.X = np.concatenate((self.X, Xaug[shuffle][:n_generated_samples]), 0)
        self.Y = np.concatenate((self.Y, Yaug[shuffle][:n_generated_samples]), 0)                       
                                
        return self.X, self.Y    
    
    def noise(self, var=0.05):

        self.X = self.X + np.random.normal(np.nanmean(self.X), var, size=self.X.shape)

        return self.X  
    
def calculate_distance(color1, color2):
    """Calculate Euclidean distance between two RGB colors."""
    return np.linalg.norm(np.array(color1) - np.array(color2))

def generate_random_color(seed=None):
    """Generate a random RGB color."""
    np.random.seed(seed)
    return tuple(np.random.rand(3))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hexadecimal color code."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

def gen_colorscale(n, dmin, cmap=False, seed=None):
    """Create a colormap with n colors, ensuring minimum distance dmin."""
    colors = [generate_random_color(seed)]

    while len(colors) < n:
        new_color = generate_random_color()

        # Check the distance to existing colors
        distances = [calculate_distance(new_color, existing_color) for existing_color in colors]
        min_distance = min(distances)

        if min_distance >= dmin:
            colors.append(new_color)
            
    if cmap:
        return ListedColormap(colors)
    else:
        return [rgb_to_hex(c) for c in colors]
    
