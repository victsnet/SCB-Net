import numpy as np
import rioxarray
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.interpolate import griddata
import rasterio
import warnings

def normalize(x, lim=255.):
    return (x-np.min(x))/(np.max(x)-np.min(x))*lim


def remove_outliers(var_array, printit=True):
    # interquantile range
    Q1 = np.percentile(var_array, 25) 
    Q3 = np.percentile(var_array, 75) 
    IQR = Q3 - Q1
    
    if printit:
        print(f'with outliers - min:{np.min(var_array):.3f} / max:{np.max(var_array):.3f}')

    # superior limite
    RPS = np.percentile(var_array, 95) # 95th percentile
    var_array = np.where(var_array > Q3 + 1.5*IQR, RPS, var_array)
    
    # inferior limite
    RPI = np.percentile(var_array, 5) # 5th percentile
    var_array = np.where(var_array < Q1 - 1.5*IQR, RPI, var_array)
    
    if printit:
        print(f'without outliers - min:{np.min(var_array):.3f} / max:{np.max(var_array):.3f}')
    
    return var_array

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


def predict_batches(model, data, dim, classes):
    
    '''
    args:
        model - DL model
        data - dataset
        dim - batches' dimensions  
        classes - array of classes
    output:
        result
    '''
    from sys import stdout
    (y_max, x_max, _) = data.shape
    result            = np.zeros((y_max, x_max, classes.size))
    batch             = np.zeros((y_max-dim[1], *dim))

    m = 0
    n = 0
    for x in range(dim[0]//2,x_max-dim[0]//2): 
        stdout.write("\r%d" % m)
        stdout.flush()
        m = m + 100/(x_max-dim[0])

        for y in range(dim[1]//2,y_max-dim[1]//2):
            batch[n,:,:,:] = data[y-dim[1]//2:y+dim[1]//2, x-dim[0]//2:x+dim[0]//2, :] 
            n = n + 1

        result[dim[1]//2:y_max-dim[1]//2,x,:] = model.predict_on_batch(batch)[:, :classes.size]
        n = 0
    
    return result


def get_batches_masks(image, mask, dim, patch_num, max_it=10000, dtype=np.float32):
    
    '''
    image - input array data
    mask - labelled array data
    dim - batches' dimensions
    patch_num - number of samples
    
    output:
            X, Y, y
    '''
    from numpy.random import choice
    import time
    
    # labels in mask
    labels = np.unique(mask)
    labels = labels[np.where(np.isnan(labels) == False)]
    
    # select only labelled pixels
    size = int(patch_num) 
    # create grids to store values
    X = np.zeros((size, *dim), dtype=dtype)
    Ym = np.full((size, dim[0], dim[1], 1), fill_value=np.nan, dtype=np.uint8)
    
    # store central pixel labels
    y = []

    # count images
    i = 0

    # select non-nan values
    idy, idx = np.where(np.isnan(mask) == False)
    #mask = np.where(np.isnan(mask) == True, 0, mask)
    elems = np.arange(0, idy.size, dtype=int)
            
    count = 0 
    iteration = 0
    
    tic = time.time()
    while count < size:
            
        # create batches
        e = choice(elems, size=1, replace=False)
        # create subset
        iy, ix = int(idy[e]), int(idx[e])
        
        # submask
        msk = mask[iy-dim[0]//2:iy+dim[0]//2, ix-dim[1]//2:ix+dim[1]//2]

        img = image[iy-dim[0]//2:iy+dim[0]//2, ix-dim[1]//2:ix+dim[1]//2, :]
        dimm = img.shape
                
        if(dimm == dim):
            X[i] = img
            Ym[i:, :, :, 0] = msk
            y.append(mask[iy, ix])
            # avoid repeting central pixel
            elems = np.delete(elems, e)
            count += 1
            i += 1
        else:
            elems = np.delete(elems, e)     
            
        # count iterations
        iteration += 1
        # to avoid infinity loop
        if iteration > max_it:
            # force it to stop
            count = size
            
    toc = time.time()
    
    print(f'Computing time: {((toc-tic)/60.):.2f} min.')
            
    return X[:i], Ym[:i], y


def subset_infos(df, column, threshold=5):
    
    labels, counts = np.unique(df[column], return_counts=True)
    proportion = (counts/np.sum(counts))*100
    sub_labels = np.where(proportion > threshold)[0] # labels that present more than X% of the samples.
    
    # conditions --> indexes
    indexes = [] 
    for n in range(sub_labels.size):
        indexes.append(df[df[column] == labels[sub_labels[n]]].index)    
    
    # remove possible duplicates
    indexes = np.unique(np.concatenate(indexes))
    
    
    # define new proportions based on sub-labels
    proportions = proportion[sub_labels]/np.sum(proportion[sub_labels]) # make selected labels sum up 100%
    
    return indexes, labels[sub_labels], proportions, counts[sub_labels]


def create_val_set(subset, sublabels, div=20, seed=13, return_original=False):

    validation = []

    np.random.seed(seed)

    for label in sublabels:
        label_index = np.array(subset[subset.code_r2 == label].index, dtype=np.int16)

        if div == False:
            validation.append(np.random.choice(label_index, size=1, replace=False))

        else:
            validation.append(np.random.choice(label_index, size=label_index.size//div, replace=False))

    validation = np.concatenate(validation)

    val_df = subset.loc[validation].copy()
    val_df.reset_index(drop=True, inplace=True)

    # "train" set
    set_index = np.setdiff1d(np.arange(subset.shape[0]), validation)
    df = subset.loc[set_index].copy()
    if return_original:
        df = subset.copy()
    df.reset_index(drop=True, inplace=True)
    
    return df, val_df


def merge_xarrays(paths, scaling):
    
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

def bounds2cells(df=None, pixel_size=None, block_size=1, x=None, y=None, check_intersection=False):

    # Create an empty list to store grid polygons
    grid_polygons = []

    # Calculate the bounds of the extent
    block_size = float(block_size)
    if df is not None:
        xmin, ymin, xmax, ymax = df.total_bounds
        Lx = Ly = pixel_size * block_size
    if (x is not None) and (y is not None):
        xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max() 
        Lx, Ly = np.abs(np.mean(np.diff(x, axis=1))), np.abs(np.mean(np.diff(y, axis=0)))
        Lx *= block_size
        Ly *= block_size 
    xc = np.arange(xmin, xmax, Lx)
    yc = np.arange(ymin, ymax, Ly)
    ny, nx = yc.size, xc.size        
    # Create a loop to generate grid polygons
    for yi in yc:
        for xi in xc:
            # Create a polygon for each cell
            cell = Polygon([
                (xi, yi),
                (xi + Lx, yi),
                (xi + Lx, yi + Ly),
                (xi, yi + Ly)
            ])
            
            # Check for intersection with existing polygons
            if check_intersection:
                for existing_poly in grid_polygons:
                    if cell.intersects(existing_poly):
                        # If intersection, adjust cell boundaries
                        cell = cell.difference(existing_poly)
                    
            # append cell to list
            grid_polygons.append(cell)
            
    # Create a GeoDataFrame from the grid polygons
    grid_gdf = gpd.GeoDataFrame(grid_polygons, columns=['geometry'])

    # Set the coordinate reference system (CRS) of the grid to match the extent's CRS
    grid_gdf.crs = df.crs
    coords = np.array([(polyg.centroid.x, polyg.centroid.y) for polyg in grid_gdf.geometry])
    xc = coords[:, 0].reshape((ny, nx))
    yc = coords[:, 1].reshape((ny, nx))
    
    return grid_gdf, (xc, yc)

def spatial_blocks_mode(spatial_blocks, gdf, column):
    
    samples_gdf = gpd.sjoin(gdf, spatial_blocks,
                                     how='inner', predicate='intersects')

    unique_index = np.unique(samples_gdf.index_right)
    labels = np.full(spatial_blocks.shape[0], 'XXXX')

    code_md = []
    for uq in unique_index:
        code_md.append(str(samples_gdf[samples_gdf.index_right == uq][column].mode()[0]))

    labels[unique_index] = code_md
    spatial_blocks['labels'] = labels
    
    return spatial_blocks


def stratified_spatial_split(spatial_blocks, gdf, column, labels, fraction=0.2):
    
    # run for all selected lito_codes
    train_index = []
    val_index = []
    label_train_index = []
    label_val_index = []
    
    # gets the mode per block
    spatial_blocks = spatial_blocks_mode(spatial_blocks, gdf)
    
    for label in labels:
        
        label_index = np.unique(spatial_blocks[spatial_blocks.labels == label].index)
        size = int(label_index.size * fraction)
        label_val_idx = np.random.choice(label_index, size=size, replace=False)
        label_train_idx = np.setdiff1d(label_index, label_val_idx)        
        
        
        # get only the grids which intersect samples
        train_samples_gdf = gpd.sjoin(spatial_blocks.iloc[label_train_idx], gdf[gdf[column] == label],
                                     how='inner', predicate='intersects')
        
        val_samples_gdf = gpd.sjoin(spatial_blocks.iloc[label_val_idx], gdf[gdf[column] == label],
                                     how='inner', predicate='intersects')
        
        # get unique idx
        train_idx = np.unique(train_samples_gdf.index_right)
        val_idx = np.unique(val_samples_gdf.index_right)
        
        # append
        train_index.append(train_idx)
        val_index.append(val_idx)
        label_train_index.append(label_train_idx)
        label_val_index.append(label_val_idx)        
        
    # concatenate lists of index
    train_index = np.concatenate(train_index)
    val_index = np.concatenate(val_index)
    label_train_index = np.concatenate(label_train_index)
    label_val_index = np.concatenate(label_val_index)
    
    # subset
    subsets_col = np.full(spatial_blocks.shape[0], 0)
    subsets_col[label_train_index] = 1
    subsets_col[label_val_index] = 2
    spatial_blocks['subset'] = subsets_col
    
    return (spatial_blocks, spatial_blocks.iloc[label_train_index], spatial_blocks.iloc[label_val_index],
            train_index, val_index)

def spatial_blocks_probability(spatial_blocks, gdf, column, labels, bf=None):
    
    if bf is not None:
        spatial_blocks['geometry'] = spatial_blocks.geometry.buffer(bf)
    
    n_classes = len(labels)
    samples_gdf = gpd.sjoin(gdf, spatial_blocks, how='inner', predicate='intersects')

    unique_index = np.unique(samples_gdf.index_right)
    prob_array = np.full((spatial_blocks.shape[0], n_classes), -9999., dtype=np.float32)
    
    prob_list = []
    for uq in tqdm(unique_index):
        # calculate class probability
        class_prob = np.zeros((n_classes,), dtype=np.float32)
        subset = samples_gdf[samples_gdf.index_right == uq][column]
        local_labels, local_counts = np.unique(subset, return_counts=True)
        
        # Use numpy.isin to check if each element of B is in A
        bool_pos = np.isin(labels, local_labels)
        if bool_pos.any():
            # calculate the local probability per class
            local_prob = local_counts/np.sum(local_counts)
            # add to class_prob
            class_prob[bool_pos] = local_prob
        prob_array[uq] = class_prob
        
    for n, label in enumerate(labels):
        spatial_blocks[f'prob_{label}'] = prob_array[:, n]
    
    return spatial_blocks

def interp_column(gdf_A, gdf_B, column, crs):
    
    # Ignore the specific UserWarning about geographic CRS
    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.", category=UserWarning)
    
    gdf_A = gdf_A.to_crs(crs)
    gdf_B = gdf_B.to_crs(crs)

    x_a, y_a = gdf_A.geometry.centroid.x, gdf_A.geometry.centroid.y
    x_b, y_b = gdf_B.geometry.centroid.x, gdf_B.geometry.centroid.y

    # Interpolate values from gdf_B to gdf_A based on their centroids
    subset_interpolated = griddata((x_b, y_b), gdf_B[column], (x_a, y_a), method='nearest')

    # Add the interpolated values to gdf_A
    gdf_A[column] = subset_interpolated
    
    return gdf_A

def export_tif(ds, crs, path, x='lon', y='lat'):

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