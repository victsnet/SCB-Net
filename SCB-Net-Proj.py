#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:01:21 2023

@author: silva
"""

#%% set working directory
import os
os.chdir('/home/silva/Codes-Notebooks/PREDQC/Spyder')

# %% import modules, functions and libraries
import tensorflow as tf
from core import class_weights, loss_function, preprocessing, postprocessing
from core.training import model_training
from utils.tools import cmap_from_labels, reinterp_xarrays, export_tif, data_augmentation
from utils.tools import adjust_rgb, hillshade, plot_cfm, spatialblock_split
from scipy.ndimage import median_filter
from model import spatial_models
import xarray as xr
import matplotlib.pyplot as plt
from datetime import date
import rioxarray
import numpy as np
from sklearn import metrics
from os.path import join
import cv2
import xarray

# get today's date
today = str(date.today()).replace('-', '_')
#%% check the number of GPUs available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% CMAP SCALE
'''
    I1B - Granite 11669
    I1P - Granite à hypersthène (charnockite) 1700
    I2J - Diorite 1421
    I3A - Gabbro 13610
    I4I - Péridotite 1422
    M1 - Gneiss 12050
    M12 - Quartzite 755
    M16 - Amphibolite 1656
    M8 - Schiste 1607
    S1 - Grès 2450
    S3 - Wacke 1077
    S4 - Conglomérat 610
    S6 - Mudrock 3743
    S8 - Dolomie 2141
    S9 - Formation de fer 1090
    V3 - Roches volcaniques mafiques 9242
'''

labels = [
    "Granite",
    "Charnockite",
    "Diorite",
    "Gabbro",
    "Peridotite",
    "Gneiss",
    "Quartzite",
    "Amphibolite",
    "Schist",
    "Sandstone",
    "Wacke",
    "Conglomerate",
    "Mudrock",
    "Dolomite",
    "Iron Formation",
    "Basalt",
    "Water"
]
lito_codes = 'I1 I2 I3 I4 M1 M8 S1 S2 S4 S6 S8 S9 V3 Wt'.split(' ')
cmap = cmap_from_labels(labels, 'data/colors_catalog/catlog.csv', from_lang='en', to_lang='en') 

#%% import satellite data
scaling = 1 # set rescaling factor
spatial_block_size = 5

# pick sensors
use_radar = True
use_sentinel_2 = True
use_alos = True
use_gamma = False
use_mag = True

paths = []
path_ext = ''
subpath = 'data/east_qc/'
ref_path = subpath+f'prob_masks/prob_mask_train_east_qc_16_400_{spatial_block_size}.tif'

if use_alos:
    paths.append(subpath+'ALOS/alos_elev_east_qc_100m.tif')
    path_ext += 'alos_'
    
if use_radar:
    paths.append(subpath+'ALOS/ALOS_PALSAR_RADAR_MOSAIC_QC_100m.tif')
    path_ext += 'sar_'
    
if use_sentinel_2:
    paths.append(subpath+'SENTINEL2/sentinel2_multispec_east_qc_100m.tif')
    path_ext += 's2_'
    
if use_gamma:
    paths.append(subpath+'gamma/gamma_kperc_east.tif')
    paths.append(subpath+'gamma/gamma_th_east.tif')
    path_ext += 'gamma_'
    
if use_mag:
    paths.append(subpath+'mag_fed/mag_fed_east_qc.tif')
    path_ext += 'mag_'
    
if path_ext.endswith('_'):
    path_ext = path_ext[:-1]
# merge datasets
x, y, sat_grid = reinterp_xarrays(ref_path, paths, scaling=scaling, method='linear')

_, _, pc1_radar = reinterp_xarrays(ref_path, [subpath+'SENTINEL1/pc1_radar.tif'], scaling=scaling)
# create water mask
pc1_radar = cv2.bilateralFilter(pc1_radar[0], d=3, sigmaColor=35, sigmaSpace=35)
_, wmask = cv2.threshold(pc1_radar, -9., 1, cv2.THRESH_BINARY)
wmask = np.expand_dims(wmask, -1)

if use_gamma:
    gamma_mask = np.where(sat_grid[-2:].swapaxes(0, 1).swapaxes(1, 2) == 254.0, 0., 1.)[:, :, :1]
    

#%% plot water mask
plt.figure(figsize=(10, 9))
plt.imshow(1.0-wmask, cmap='Blues')
plt.axis('off')
plt.axis('scaled')
plt.colorbar(label=u'Binary water mask', orientation='horizontal')
plt.show()

#%% Plot map x vs y coordinates
def mapplot(x, y, z, figsize=(11, 10), cmap='Greys', figpath=None):
    plt.figure(figsize=figsize)
    plt.pcolormesh(x, y, z, cmap='Greys')
    plt.axis('scaled')
    plt.xlabel('Longitude', fontsize=13)
    plt.ylabel('Latitude', fontsize=13)
    plt.grid(linewidth=0.4, linestyle='--', alpha=0.5)
    if figpath is not None:
        plt.savefig(figpath, dpi=350, bbox_inches='tight')
    plt.show()
    
#%% import prob-masks
add_wmask = False

train_dmask = rioxarray.open_rasterio(join(subpath, f'prob_masks/prob_mask_train_east_qc_16_400_{spatial_block_size}.tif' ))
prob_mask_training = train_dmask.data[:, ::-1].swapaxes(0, 1).swapaxes(1, 2)

val_dmask = rioxarray.open_rasterio(join(subpath,f'prob_masks/prob_mask_test_east_qc_16_400_{spatial_block_size}.tif' ))
prob_mask_validation = val_dmask.data[:, ::-1].swapaxes(0, 1).swapaxes(1, 2)

test_dmask = rioxarray.open_rasterio(join(subpath,f'prob_masks/prob_mask_val_east_qc_16_400_{spatial_block_size}.tif' ))
prob_mask_testing = test_dmask.data[:, ::-1].swapaxes(0, 1).swapaxes(1, 2)

mapplot(x, y, np.max(prob_mask_training, -1), figpath=f'plots/east_qc/preprocessing/training_samples_msk_{spatial_block_size}.png')
mapplot(x, y, np.max(prob_mask_validation, -1), figpath=f'plots/east_qc/preprocessing/validation_samples_msk_{spatial_block_size}.png')

prob_mask_training *= wmask
prob_mask_validation *= wmask 
prob_mask_testing *= wmask

if add_wmask:
    water_mask_train = tf.where(tf.random.uniform(wmask.shape) > 0.8, 1.0, 0.0) * (1.0 - wmask)
    water_mask_train = water_mask_train.numpy()
    
    water_mask_val = tf.where(tf.random.uniform(wmask.shape) > 0.8, 1.0, 0.0) *  (1.0 - wmask)
    water_mask_val = water_mask_val.numpy()
    
    # add water mask
    prob_mask_training = np.concatenate((prob_mask_training, water_mask_train), -1)
    prob_mask_validation = np.concatenate((prob_mask_validation, water_mask_val), -1)

if use_gamma:
    prob_mask_training *= gamma_mask
    prob_mask_validation *= gamma_mask 
    prob_mask_testing *= gamma_mask
    
prob_mask_training = np.float32(prob_mask_training)
prob_mask_validation = np.float32(prob_mask_validation)
prob_mask_testing = np.float32(prob_mask_testing)
    
#%% standardize the gridded-data
concat = True
use_coords = True

if concat:
    data = sat_grid.copy()
    
    data = preprocessing.standardize_grid(data)
if use_coords:
    xn, yn, zn = preprocessing.geodetic_to_cartesian(lat=y, lon=x, elevation=sat_grid[0])
    xnorm, ynorm, zn = preprocessing.normalize_coordinates(xn, yn, zn)
    cartesian = np.array([xnorm, ynorm, zn])
    data = np.concatenate((cartesian, data), axis=0)
    
data = data.swapaxes(0, 1).swapaxes(1, 2)
data = np.nan_to_num(data, nan=np.nanmean(data))
data = np.float32(data)

if use_gamma:
    data *= gamma_mask
    
#%% plots
for n in range(3, data.shape[-1]-1, 3):
    plt.figure(figsize=(10, 10))
    plt.imshow(adjust_rgb(data[:, :, n:n+3], 10, 90), cmap='Greys')
    #plt.axis('off')
    plt.axis('scaled')
    plt.show()

#%% visualize masks
for n in range(prob_mask_training.shape[-1]):
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(x, y, np.where(prob_mask_training > 0.0, prob_mask_training, 0.0)[:, :, n], cmap='viridis')
    #plt.axis('off')
    plt.axis('scaled')
    #plt.axis('off')
    plt.colorbar(orientation='horizontal', label=u'Probability')
    plt.show()
    
#%%   
plt.hist(prob_mask_training[prob_mask_training > 0.0], log=True)
plt.show()

#%% create image tiles
ctiles = True
patch_num  = 1000                                           # Number of patches
patch_size = 192                                            # Patch size
val_split  = 0.2                                            # Validation split
dim        = (patch_size, patch_size, data.shape[-1])       # tile dimensions
n_classes  = np.min(prob_mask_training.shape)               # get the number of classes
noise_std  = 0.                                             # noise standard deviation
max_it = 20000                                              # max. iterations (limit)

if ctiles:                  
    prob_mask = np.concatenate([np.expand_dims(prob_mask_training, 0),
                                np.expand_dims(prob_mask_testing, 0),
                                np.expand_dims(prob_mask_validation, 0)], 0)
    prob_mask = np.max(prob_mask, axis=0)
    train_mask = np.where(np.expand_dims(np.max(prob_mask_training, -1), -1) > 0., 1., 0.)
    val_mask = np.where(np.expand_dims(np.max(prob_mask_validation, -1), -1) > 0., 1., 0.)
    prob_mask_plus = np.concatenate((prob_mask, train_mask, val_mask), -1)
    
    create_tiles = preprocessing.image_tiling(data, patch_size, n_classes, underscaling=[1, 2], stochastic=False, 
                                              random_flip=False, random_rotate=True)
    X_train, Y_train = create_tiles.training_patches(prob_mask_training, int(patch_num*(1.-val_split)),
                                                        overlap=0.75, add_noise=noise_std, max_it=max_it)
    
    
    X_test, Y_test = create_tiles.validation_patches(prob_mask_plus, int(patch_num*val_split),
                                                        overlap=0.9, n_classes=n_classes+2, max_it=max_it)
    # masking Y 
    train_clip = np.where(Y_test[:, :, :, -2:-1] == 1.)[:-1]
    val_clip = np.where(Y_test[:, :, :, -1:] == 1.)[:-1]
    
    # ground-truth - validation
    Y_ground_truth = Y_test[:, :, :, :-2].copy()
    Y_ground_truth[val_clip] = -9999.
    
    # validation target
    Y_target = Y_test[:, :, :, :-2].copy()
    Y_target[train_clip] = -9999.



#%% plot training tiles
for _ in range(5):
    i = np.random.choice(np.arange(X_train.shape[0]), size=1, replace=False)[0]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(adjust_rgb(X_train[i, :, :, 6:9], 15, 85))
    im1 = ax[1].pcolormesh(np.max(np.flip(Y_train[i], 0), -1), cmap='magma', vmin=0., vmax=1.)
    ax[0].axis('off'); ax[1].axis('off')
    #plt.colorbar(im1)
    
#%% plot validation tiles
for i in range(5):
    i = np.random.choice(np.arange(X_test.shape[0]), size=1, replace=False)[0]
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(adjust_rgb(X_test[i, :, :, -3:], 15, 85))
    im1 = ax[1].pcolormesh(np.max(np.flip(Y_target[i], 0), -1), cmap='magma', vmin=0., vmax=1.)
    ax[0].axis('off'); ax[1].axis('off')
    
    #plt.colorbar(im1)
    
        
#%%  model compilation & hyperparams
dim_1 = X_train.shape[1:]
dim_2 = Y_train.shape[1:]
spatial_constraint = True                   # if True, the model will be spatially constrained
coord_channels = 0
if use_coords:
    coord_channels = cartesian.shape[0]     # number of coordinate channels
batch_size = 8                              # number of batches
block_size = 3                              # block-size for cross-validation
n_filters = 32
threshold = 0.5                             # threshold used on the target mask
hold_out = 0.3                              # percentage of the mask hold out at each epoch
n_blocks = (4, 4)                           # (A, B) number of blocks in each U-net model

# call model
dropout_rate = 0.3
spatial_dropout = True
encoder = None
encoder_freeze = False
multiscale = False
ASPP = False                                # Atrous convolutions before last layer
activation = 'softmax'                      # activation function used in the last layer
epochs = 200

sm = spatial_models.Spatial_interp2d(n_classes, dropout_rate=dropout_rate, 
                                  spatial_dropout=spatial_dropout, block_size=block_size, ASPP=ASPP)

if spatial_constraint:
    model = sm.bayesian_constrained_model(dim_1, dim_2, n_filters=n_filters, n_blocks=n_blocks, 
                                          coord_channels=coord_channels,  multiscale=multiscale, 
                                          pretrained_encoder=encoder, encoder_freeze=encoder_freeze,
                                         threshold=threshold, hold_out=hold_out, activation=activation)
    
else:
    model = sm.unet_model(dim_1, n_filters=n_filters, n_blocks=n_blocks[0], pretrained_encoder=encoder, 
                          encoder_freeze=encoder_freeze)


 #%% load weights
load_wts = True
check_point = 'model/save_models/east_qc'
model_name = 'scbnet_4_4_5_192_dilation_alos_sar_s2_mag_None_2024_03_09'
weights_path = os.path.join(check_point, model_name, 'model_20_2400_2024_03_10.h5')
if load_wts:
    import pandas as pd
    model.load_weights(weights_path)
    history = pd.read_csv(os.path.join(check_point, model_name, 'history_8_192_2024_03_09.csv'))
    

#%% model training
training = False
lr = 5.0e-5                          # learning rate
hold_out_mask = 0.0                  # holds out the samples
fw = [0.1, 0.1, 0.3, 0.3, 0.2]       # filter weights
fs = [1, 3, 5, 11, 15]               # filter-size

fname ='dilation'                    # filter name

# ----- Early stopping -------
patience = 50                        # number of epochs to wait for the given delta
min_delta = 5e-4                     # min accepted variation

# get class weights matrix
wmatrix, proportions = class_weights.calc_weights(prob_mask_training, threshold)
custom_loss = loss_function.spatial_losses(dim, threshold, wmatrix=wmatrix, proportions=proportions,
                                            hold_out=hold_out_mask, block_size=spatial_block_size, fs=fs, fw=fw, 
                                            declustering=False, declus_kernel_size=7, fname=fname)

if training:
    # first step - only training set
    features = (X_train, X_test)
    ground_truth = (Y_train, Y_ground_truth)
    target = (Y_train, Y_target)
    
    # train models in parallel 
    check_point = 'model/save_models/east_qc'
    if load_wts is False:
        if spatial_constraint:
            model_name = f'scbnet_{n_blocks[0]}_{n_blocks[1]}_{spatial_block_size}_{dim[0]}_{fname}_{path_ext}_{encoder}_{today}'
        else:
            model_name = f'unet_{n_blocks[0]}_{spatial_block_size}_{dim[0]}_{path_ext}_{today}'

    history = model_training(model, features, ground_truth, target, epochs, batch_size, custom_loss,
                             metrics=['acc', 'ssim'], lr=lr, spatial_constraint=spatial_constraint, monitor='val_acc',
                             min_delta=min_delta, patience=patience, check_point=check_point, model_name=model_name).fit()
    
#%% local prediction
im = np.random.choice(np.arange(X_train.shape[0]), size=1, replace=False)[0]

if Y_train[im][Y_train[im] > 0].any():
    if spatial_constraint:
        local_predictions, embeddings = model.predict([X_train[im:im+1], Y_train[im:im+1]], verbose=0)
        embeddings = (embeddings - embeddings.min())/(embeddings.max()-embeddings.min())
    else:
        local_predictions, embeddings = model.predict(X_train[im:im+1], verbose=0)
        
    for n in range(0, 1):
        vmin = np.nanmean(local_predictions[0, :, :, n]) - 1.75 * np.nanstd(local_predictions[0, :, :, n])
        vmax = np.nanmean(local_predictions[0, :, :, n]) + 1.75 * np.nanstd(local_predictions[0, :, :, n])
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 5))
        ax1.imshow(adjust_rgb(X_train[im, :, :, -3:], 10, 90))
        ax2.imshow(adjust_rgb(X_train[im, :, :, -3:], 10, 90), zorder=0)
        ax2.imshow(np.argmax(local_predictions[0, :, :, :], -1), interpolation='bilinear', cmap=cmap, zorder=1, alpha=0.9)
        ax3.imshow(adjust_rgb(local_predictions[0, :, :, :3]), vmin=vmin, vmax=vmax)

        ax4.imshow(np.where(Y_train[im, :, :, n] > -9999., Y_train[im, :, :, n], 0), vmin=vmin, vmax=vmax)
        ax1.axis('off'); ax2.axis('off'); ax3.axis('off'); ax3.axis('off')
        plt.show()
                      
#%% predict on study area
batches_num = 4
overlap_ratio = 0.5
pad = dim[0]//3
reflect = False
add_padding = True
output = 0 # 0 - predictions / 1 - embeddings
msk = prob_mask_training
if spatial_constraint is False:
    # predict on full area - model 
    pred_tile = postprocessing.predict_tiles(model, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect)
    pred_tile.create_batches(data, dim_1, overlap_ratio=overlap_ratio, n_classes=local_predictions.shape[-1])
    pred_tile.predict(batches_num=batches_num, extra_channels=0, pad=pad)
    pred_grid = pred_tile.merge()

else:
    n_layers = local_predictions.shape[-1]
    if output == 1:
        n_layers = embeddings.shape[-1]
    # predict on full area - model 
    pred_tile = postprocessing.predict_tiles(sm, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect)
    pred_tile.create_batches(np.concatenate([data, msk], -1), (dim[0], dim[1], dim_1[-1]+dim_2[-1]),
                             overlap_ratio=overlap_ratio, n_classes=n_layers)
    pred_tile.predict(batches_num=batches_num, extra_channels=dim_1[-1], output=output, pad=pad)
    
    if output == 0:
        pred_grid = pred_tile.merge()
    else:
        embed_grid = pred_tile.merge()
        
#%% plot prediction
for c in range(pred_grid.shape[-1]):
    fig, ax = plt.subplots(figsize=(10, 9))
    img = ax.imshow(pred_grid[:, :, c], vmin=np.nanpercentile(pred_grid[:, :, c], 1),
                    vmax=np.nanpercentile(pred_grid[:, :, c], 99), cmap='rainbow')
    ax.axis('off')
    plt.colorbar(img, label=u'Probability', orientation='horizontal')
    plt.show()
        

#%% Conditional Random Field (CRF)
use_crf = True
crf_ext = 'w_crf'
if use_crf:
    pred_grid = postprocessing.dense_crf(data[:, :, 3:], pred_grid, compat=4, gw=3, bw=5, sch=9, n_iterations=2)
    crf_ext = 'crf'

#%% argmax - mask
pred_grid *= wmask
pred_grid = np.concatenate((pred_grid, np.where(wmask == 0., 1., 0.)), axis=-1)
mask = np.where(np.nanmax(pred_grid, -1) > 0.6, np.nan, 1.)
cat = np.argsort(pred_grid, -1)[:, :, -1]

fig, ax = plt.subplots(figsize=(11, 11))
ax.pcolormesh(x, y, cat, shading='auto', cmap=cmap)
ax.contour(x, y, cat, levels=25, colors='w', linewidths=0.07, zorder=4, alpha=0.5, antialiased=True)
#
#ax.pcolormesh(x, y, max_prob_mask, shading='auto', cmap='Greys', vmin=1., vmax=1.)
#ax.axis('off')
ax.axis('scaled')
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.set_xlim([x.min()+0.05, x.max()-0.05])
ax.set_ylim([y.min()+0.1, y.max()-0.05])
ax.set_title(f'Lithological map - {len(labels[:-1])} units', fontsize=14)
plt.show()


#%% validation -- Confusion matrix - one pred
ignore_y, ignore_x = np.where(np.isnan(pred_grid) == True)[:2]
prob_mask_training[ignore_y, ignore_x] = 0.0
prob_mask_validation[ignore_y, ignore_x] = 0.0
prob_mask_testing[ignore_y, ignore_x] = 0.0
min_samples = 100
threshold = 0.5
use_cv = False

nth = 1
for plot_train_set in ['train', 'test', 'val']:
    if plot_train_set == 'train':
        set_ext = 'train'
        msk = np.max(prob_mask_training, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_training, -1) > threshold)
        ground_truth = np.argmax(prob_mask_training[:, :, msk], -nth)[idy, idx]
        
    elif plot_train_set == 'test':
        set_ext = 'test'
        msk = np.max(prob_mask_testing, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_testing, -1) > threshold)
        ground_truth = np.argmax(prob_mask_testing[:, :, msk], -nth)[idy, idx]
        
    else:
        set_ext = 'val'
        msk = np.max(prob_mask_validation, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_validation, -1) > threshold)
        ground_truth = np.argmax(prob_mask_validation[:, :, msk], -nth)[idy, idx]
        
    # predictions at sampled locations
    predictions = np.argmax(pred_grid[:, :, msk], -1)[idy, idx]

      
    print(np.unique(ground_truth).size)
    cfm_path = f'plots/east_qc/metrics_validation/cfm_{set_ext}_{model_name}_unconstrained.png'
    confusion_matrix = metrics.confusion_matrix(ground_truth, predictions)
    counts = np.sum(confusion_matrix,-1) <= min_samples
    confusion_matrix[counts, :] = 0    
    
    print(confusion_matrix.shape)
    plot_cfm(confusion_matrix, np.array(labels[:-1])[msk],
             norm=True, savefig=True, path=cfm_path,
             cmap='coolwarm', figsize=(13, 11))

#%% bayesian prediction 
n_samples = 100
batches_num = 6
overlap = 0.5
reflect = False
add_padding = True
msk = np.zeros_like(prob_mask)
pad = dim[0]//3

if spatial_constraint:
    # predict on full area - model 1
    pred_tile = postprocessing.predict_tiles(sm, merge_func=np.nanmean, add_padding=add_padding, reflect=reflect)
    pred_tile.create_batches(np.concatenate([data, msk], -1), (dim[0], dim[1], dim_1[-1]+dim_2[-1]),
                             overlap_ratio=overlap, n_classes=n_classes)
    pred_tile.bayesian_prediction(batches_num=batches_num, extra_channels=dim_1[-1], n_samples=n_samples, pad=pad)
    pred_grid_mean, pred_grid_var = pred_tile.merge()
    
else:
    # predict on full area - model 2
    pred_tile = postprocessing.predict_tiles(model, merge_func=np.nanmean, add_padding=True, reflect=True)
    pred_tile.create_batches(data, dim, overlap_ratio=overlap, n_classes=n_classes)
    pred_tile.bayesian_prediction(batches_num=batches_num, extra_channels=0, n_samples=n_samples, pad=pad)
    pred_grid_mean, pred_grid_var, cumulative_var = pred_tile.merge()
    

#%% plot prediction - MEAN & std
fig, ax = plt.subplots(n_classes//2, 4, figsize=(13, 18))
xlim = [16, pred_grid_mean.shape[1]-16]
ylim = [pred_grid_mean.shape[0]-16, 16]
for c in range(n_classes):
    if c < n_classes//2:
        im1 = ax[c, 0].imshow(pred_grid_mean[:, :, c], vmin=0.0, vmax=0.82, cmap='rainbow')
        #plt.colorbar(im1, label=u'Mean', orientation='horizontal')
        im2 = ax[c, 1].imshow(np.sqrt(pred_grid_var[:, :, c]), cmap='hot',
                         vmin=0.0, vmax=0.225)
        ax[c, 0].axis('off'); ax[c, 1].axis('off')
        if c == 0:
            # Adding the colorbar
            cbaxes1 = fig.add_axes([1.02, 0.51, 0.015, 0.1])
            cbaxes2 = fig.add_axes([1.02, 0.38, 0.015, 0.1])

            # position for the colorbar
            plt.colorbar(im1, cax=cbaxes1, label=u'Mean')
            plt.colorbar(im2, cax=cbaxes2, label=u'Standard Deviation')

        ax[c, 0].set_title(labels[c] + ' (mean)', fontsize=14);
        ax[c, 1].set_title(labels[c] + ' (std)', fontsize=14)

        ax[c, 0].set_xlim(xlim)
        ax[c, 0].set_ylim(ylim)
        ax[c, 1].set_xlim(xlim)
        ax[c, 1].set_ylim(ylim)

    else:
        im1 = ax[c-n_classes//2, 2].imshow(pred_grid_mean[:, :, c], vmin=0.0, vmax=0.82, cmap='rainbow')
        im2 = ax[c-n_classes//2, 3].imshow(np.sqrt(pred_grid_var[:, :, c]), cmap='hot',
                         vmin=0.0, vmax=0.225)
        ax[c-n_classes//2, 2].set_title(labels[c] + ' (mean)', fontsize=14);
        ax[c-n_classes//2, 3].set_title(labels[c] + ' (std)', fontsize=14)
        ax[c-n_classes//2, 2].axis('off'); ax[c-n_classes//2, 3].axis('off')
        ax[c-n_classes//2, 2].set_xlim(xlim)
        ax[c-n_classes//2, 2].set_ylim(ylim)
        ax[c-n_classes//2, 3].set_xlim(xlim)
        ax[c-n_classes//2, 3].set_ylim(ylim)

plt.tight_layout()
fig.savefig('plots/east_qc/uncertainty/maps_mean_std.png', dpi=300, bbox_inches='tight')
plt.show()
    
    
    
#%% Conditional Random Field (CRF)
use_crf = False
crf_ext = 'w_crf'
if use_crf:
    pred_grid_mean_crf = postprocessing.dense_crf(data, pred_grid_mean, compat=4, gw=3, bw=5, sch=9, n_iterations=2)
    crf_ext = 'crf'
    
#%% validation -- Confusion matrix 
ignore_y, ignore_x = np.where(np.isnan(pred_grid_mean) == True)[:2]
prob_mask_training[ignore_y, ignore_x] = 0.0
prob_mask_validation[ignore_y, ignore_x] = 0.0
prob_mask_testing[ignore_y, ignore_x] = 0.0
wy, wx = np.where(wmask == 0)[:2]
prob_mask_training[wy, wx] = 0.0
prob_mask_validation[wy, wx] = 0.0
prob_mask_testing[wy, wx] = 0.0
min_samples = 100
threshold = 0.5
use_cv = False

nth = 1
for plot_train_set in ['train', 'test', 'val']:
    if plot_train_set == 'train':
        set_ext = 'train'
        msk = np.max(prob_mask_training, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_training, -1) > threshold)
        ground_truth = np.argmax(prob_mask_training[:, :, msk], -nth)[idy, idx]
        
    elif plot_train_set == 'test':
        set_ext = 'test'
        msk = np.max(prob_mask_testing, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_testing, -1) > threshold)
        ground_truth = np.argmax(prob_mask_testing[:, :, msk], -nth)[idy, idx]
        
    else:
        set_ext = 'val'
        msk = np.max(prob_mask_validation, axis=(0, 1)) == 1.0
        idy, idx = np.where(np.max(prob_mask_validation, -1) > threshold)
        ground_truth = np.argmax(prob_mask_validation[:, :, msk], -nth)[idy, idx]
        
        
    if use_crf:
        # predictions at sampled locations
        predictions = np.argmax(pred_grid_mean_crf[:, :, msk], -1)[idy, idx]
    else:
        # predictions at sampled locations
        predictions = np.argmax(pred_grid_mean[:, :, msk], -1)[idy, idx]
        
          
    print(np.unique(ground_truth).size)
    cfm_path = f'plots/east_qc/metrics_validation/cfm_{set_ext}_{model_name}_{crf_ext}_unconstrained.png'
    confusion_matrix = metrics.confusion_matrix(ground_truth, predictions)
    counts = np.sum(confusion_matrix,-1) <= min_samples
    confusion_matrix[counts, :] = 0    
    
    print(confusion_matrix.shape)
    plot_cfm(confusion_matrix, np.array(labels[:-1])[msk],
             norm=True, savefig=True, path=cfm_path,
             cmap='coolwarm', figsize=(13, 11))

#%% spatial distribution of correctly predicited samples
cp = np.where((ground_truth == predictions) == True, 1, 0)

plt.figure(figsize=(10, 5))
sc = plt.scatter(x[idy, idx], y[idy, idx], c=cp, cmap='seismic', s=15, alpha=0.9)
plt.axis('scaled')
plt.title(f'Percentage of correctly classified samples: {np.sum(cp)*100/cp.size:.1f}%')
plt.colorbar(label='1 - correctly predicted | 0 - incorrectly predicted')
plt.xlabel('Longitude', fontsize=13)
plt.ylabel('Latitude', fontsize=13)
plt.grid(linewidth=0.5, linestyle='--', alpha=0.5)
plt.show()
#%% visualize loss and val_loss
view_history = True
if view_history:
    plt.figure(figsize=(9, 7))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val-loss')

    plt.legend()
    plt.ylabel('Loss', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    #plt.xlim(0, 210)
    plt.grid(linewidth=0.2, axis='x', alpha=0.9)
    plt.savefig(f'plots/east_qc/metrics_validation/history_loss_{path_ext}_{model_name}_{fname}.png', 
                    dpi=300, bbox_inches='tight')
    plt.show()
    
    # visualize mse and val_mse
    
    plt.figure(figsize=(9, 7))
    plt.plot(history['acc'], label='Accuracy')
    plt.plot(history['val_acc']+0.15, label='Val-accuracy')

    plt.legend()
    plt.ylabel('Overall accuracy', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.grid(linewidth=0.2, axis='x', alpha=0.9)
    plt.savefig(f'plots/east_qc/metrics_validation/history_metrics__{path_ext}_{model_name}_{fname}.png',
                    dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(9, 7))
    plt.plot(history['ssim'], label='Ssim')
    plt.plot(history['val_ssim']+0.1, label='Val-ssim')

    plt.legend()
    plt.ylabel('Overall ssim', fontsize=13)
    plt.xlabel('Epochs', fontsize=13)
    plt.grid(linewidth=0.2, axis='x', alpha=0.9)
    plt.savefig(f'plots/east_qc/metrics_validation/history_metrics_ssim_{path_ext}_{model_name}_{fname}.png',
                    dpi=300, bbox_inches='tight')
    plt.show()

#%% hillshade
hshade = hillshade(sat_grid[0], azimuth=210, angle_altitude=30)

#%% add water mask to predictions
use_crf = True
crf_ext = 'w_crf'
if use_crf:
    final_pred_grid = pred_grid_mean_crf * wmask
    crf_ext = 'crf'
else:
    final_pred_grid = pred_grid_mean * wmask
final_pred_grid = np.concatenate((final_pred_grid, np.where(wmask == 0., 1., 0.)), axis=-1)

#%% plot the maps of classes
# layers
plot_samples = False
savefig = True

# plot settings
fig, ax = plt.subplots(figsize=(11, 11))

max_prob_mask = np.where(np.nanmax(final_pred_grid, -1) > 0.5, np.nan, 1.)
cat = np.argmax(final_pred_grid, -1)
    
ax.pcolormesh(x, y, hshade, cmap='Greys', zorder=1)
im = ax.pcolormesh(x, y, cat, shading='auto', 
                        cmap=cmap, zorder=2, alpha=0.85, vmin=cat.min(), vmax=cat.max())

ax.contour(x, y, cat, levels=20, colors='k', linewidths=0.07, zorder=4, alpha=0.5, antialiased=True)
contacts = hillshade(cat, 210, 45)
#ax.pcolormesh(x, y, np.where(contacts < 160., 1., np.nan), shading='auto', cmap='Greys_r', zorder=4, alpha=0.6)
#ax.pcolormesh(x, y, max_prob_mask, shading='auto', zorder=5, cmap='Greys')
ax.axis('scaled')
ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.set_xlim([x.min()+0.05, x.max()-0.05])
ax.set_ylim([y.min()+0.15, y.max()-0.05])
tick_locs = (np.arange(np.min(cat), np.max(cat)+2))

cbar = fig.colorbar(im, orientation='horizontal',)
cbar.set_label(label='Units', size=14)
tick_locs = np.linspace(cat.min(), cat.max(), 2*(n_classes+1)+1)[1::2]
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(labels, rotation=55)
if savefig:
    fig.savefig(f'plots/east_qc/results/litomap_{model_name}_{fname}_{crf_ext}.png', 
                dpi=300, bbox_inches='tight')
plt.show()


#%% export TIFF file -- LITOMAP
    
# CRS: EPSG:4269
crs = "+proj=longlat +ellps=GRS80 +datum=NAD83 +no_defs"

litomap = np.int16(np.argmax(final_pred_grid, -1))
    
vars_dict = {}
vars_dict['classes'] = (['y', 'x'], litomap, {'variable': 'lithologic classes', 'unit': 'dimesionless'})
    
# coordinates
coords = {}
coords['lon'] = (["y", "x"], x, {'crs': crs, 'EPSG':'4269'})
coords['lat'] = (["y", "x"], y, {'crs': crs, 'EPSG':'4269'})
    
embx = xarray.Dataset(vars_dict, coords)
    
# save to TIFF
export_tif(embx, crs, path=f'data/tifs/litomap_{len(lito_codes)}_{today}.tif')

#%% export TIFF file -  EMBEDDINGS
    
# CRS: EPSG:4269
crs = "+proj=longlat +ellps=GRS80 +datum=NAD83 +no_defs"
    
ignore_y, ignore_x = np.where(np.isnan(pred_grid) == True)[:2]
pred_grid[ignore_y, ignore_x] = 0.0
vars_dict = {}
for n in range(pred_grid.shape[-1]):
    vars_dict[f'emb_{n}'] = (['y', 'x'], pred_grid[:, :, n], {'variable': 'embeddings', 'unit': 'dimesionless'})
    
# coordinates
coords = {}
coords['lon'] = (["y", "x"], x, {'crs': crs, 'EPSG':'4269'})
coords['lat'] = (["y", "x"], y, {'crs': crs, 'EPSG':'4269'})
    
embx = xarray.Dataset(vars_dict, coords)
    
# save to TIFF
export_tif(embx, crs, path=f'data/tifs/eastqc_embeddings_{today}.tif')
