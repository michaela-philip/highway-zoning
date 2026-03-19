from datetime import datetime, timedelta, timezone
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from math import sin, cos, pi  # rotating regions
from math import floor, ceil  # truncating naics codes
import rasterio
from rasterio import transform
from pathlib import Path
from scipy.ndimage import rotate, shift
import os
import glob


dataroot = 'cnn/'
outputroot = 'cnn/'

use_saved_model = True
saved_model_filename = 'cnn/bc_model1.tar'  # path to saved model for prediction

####################################################################################################
### PARAMETERS ###

# read in data and prepare lists
sample = pd.read_pickle('data/input/samplelist.pkl')
candidate_list = pd.read_pickle('data/output/cnn_candidate_list.pkl')
grid = pd.read_pickle('data/output/sample.pkl')
hwys = grid[grid['hwy'] == 1]['grid_id'].unique().tolist()
features = ['valueh', 'rent', 'distance_to_cbd', 'dist_water', 'dist_to_hwy', 'elevation', 'hwy']
normalize_features = ['valueh', 'rent', 'distance_to_cbd', 'dist_water', 'dist_to_hwy', 'elevation'] # the only features i want to demean

cell_width = 150  # cell width in meters (convert from miles)
size_potential = 4  # potential locations: num_width_potential x num_width_potential
size_padding = 4  # number of padding cells on each side of potential grid
nc = len(features)  # number of channels: 1) other grocery stores 2) other businesses
# candidates = 0
# for city in candidate_list.keys():
#     candidates += len(candidate_list[city])

obs = grid.shape[0]
BATCH_SIZE_real = 2  # regions with missing grocery store per batch
BATCH_SIZE_fill = 2  # regions with real location filled in (-> no missing) per batch
BATCH_SIZE_random = 28  # random regions (-> no missing) per batch
BATCH_SIZE = BATCH_SIZE_real + BATCH_SIZE_fill + BATCH_SIZE_random
num_batches_predict = ceil(obs / BATCH_SIZE)

frac_train_real = 1  # fraction of real regions to use for training
frac_train_random = 1  # fraction of random (unrealized) regions to use for training

use_cuda = True

curr_epoch = 0
epoch_set_seed = list()
epoch_set_seed.append(curr_epoch)
EPOCHS = 20
ITERS = 1000
NODATA = -9999.0

print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('cell_width: ' + str(round(cell_width)) + 'm')


####################################################################################################
### SAMPLE/BATCH CREATION FUNCTIONS ###
# normalize features before creating the raster
def normalize_features_per_city(grid, features, nodata=-9999.0):
    city_means = {}
    city_stds = {}

    for city in grid['city'].unique():
        city_data = grid[grid['city'] == city]
        city_means[city] = {}
        city_stds[city] = {}

        for feature in features:
            valid_mask = city_data[feature] != nodata
            if valid_mask.any():
                city_means[city][feature] = city_data.loc[valid_mask, feature].mean()
                city_stds[city][feature] = city_data.loc[valid_mask, feature].std()
            else:
                city_means[city][feature] = 0.0
                city_stds[city][feature] = 1.0

    # Normalize the grid
    for city in grid['city'].unique():
        for feature in features:
            mean = city_means[city][feature]
            std = city_stds[city][feature]
            std = std if std != 0 and not np.isnan(std) else 1.0  
            city_mask = grid['city'] == city
            grid.loc[city_mask, feature] = (grid.loc[city_mask, feature] - mean) / std

    return grid

# create raster of grid data
def gdf_to_raster(gdf, features, label, cell_width, crs=None, nodata=-9999.0):
    if crs is None:
        crs = gdf.crs
    if crs is None:
        raise ValueError("CRS must be provided either directly or through gdf.")
    
    # use centroids of each grid square
    centroids = gdf.geometry.centroid
    xs = centroids.x.values
    ys = centroids.y.values

    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / cell_width))
    height = int(np.ceil((maxy - miny) / cell_width))
    trs = rasterio.transform.from_origin(minx, maxy, cell_width, cell_width)

    if 'city' in gdf.columns and gdf['city'].dtype == object:
        cities = pd.factorize(gdf['city']) [0] # integers 0..N-1 and unique city names
        gdf = gdf.copy()
        gdf['city'] = cities.astype(float)

    # create raster bounds and output array
    bands = features + [label]
    nb = len(bands)
    out_array = np.full((nb, height, width), nodata, dtype='float32')
    cols_idx = np.floor((xs - minx) / cell_width).astype(int)
    rows_idx = np.floor((maxy - ys) / cell_width).astype(int)

    valid = (cols_idx >= 0) & (cols_idx < width) & (rows_idx >= 0) & (rows_idx < height)

    for i, col in enumerate(bands):
        if col not in gdf.columns:
            raise KeyError(f"Column '{col}' not found")
        vals = gdf[col].to_numpy(dtype='float64')
        mask = valid & ~np.isnan(vals)
        out_array[i, rows_idx[mask], cols_idx[mask]] = vals[mask].astype('float32')

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': nb,
        'crs': crs,
        'transform': trs,
        'dtype': 'float32',
        'nodata': nodata,
        'compress': 'lzw'
    }

    outpath = Path(f'data/output/{city}_rasterized.tif')
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(outpath), 'w', **profile) as dst:
        for i in range(nb):
            dst.write(out_array[i], i+1)
    print(f'rasterization complete, file saved: {outpath}')

    feature_array = out_array[:len(features), :, :]
    label_array = out_array[-1, :, :]

    return feature_array, label_array, trs

# map grid_id to (row, col) in raster
def gridid_to_rc_map(gdf, cell_width):
    centroids = gdf.geometry.centroid
    xs = centroids.x.values
    ys = centroids.y.values
    minx, miny, maxx, maxy = gdf.total_bounds
    cols_idx = np.floor((xs - minx) / cell_width).astype(int)
    rows_idx = np.floor((maxy - ys) / cell_width).astype(int)
    return {int(gid): (int(r), int(c)) for gid, r, c in zip(gdf['grid_id'].values, rows_idx, cols_idx)}

# get window patch from feature and label arrays
def extract_patch_from_arrays(feature_array, label_array, row, col, window, mirror_var=1):
    C, H, W = feature_array.shape
    pad = window // 2

    # define the relative location of the central part of the patch
    central_start = (window - size_potential) // 2
    central_end = central_start + size_potential

    # randomly assign original cell to some location in central patch
    rel_col = np.random.randint(central_start, central_end)
    rel_row = np.random.randint(central_start, central_end)
    class_idx = (rel_row - central_start) * size_potential + (rel_col - central_start)

    patch_top = row - rel_row
    patch_left = col - rel_col

    # compute raster bounds
    r_start = max(0, patch_top)
    r_end = min(H, patch_top + window)
    c_start = max(0, patch_left)
    c_end = min(W, patch_left + window)

    # compute patch indices
    pr_start = max(0, -patch_top)
    pr_end = pr_start + (r_end - r_start)
    pc_start = max(0, -patch_left)
    pc_end = pc_start + (c_end - c_start)

    # extract
    patch = np.zeros((C, window, window), dtype = np.float32)
    patch[:, pr_start:pr_end, pc_start:pc_end] = feature_array[:, r_start:r_end, c_start:c_end]
    label_patch = np.zeros((window, window), dtype=label_array.dtype)
    label_patch[pr_start:pr_end, pc_start:pc_end] = label_array[r_start:r_end, c_start:c_end]

    # randomly mirror
    if mirror_var == -1:
        patch = np.flip(patch, axis=2).copy()
        rel_col = window - 1 - rel_col  # update relative col if mirrored
        class_idx = (rel_row - central_start) * size_potential + (rel_col - central_start)
    return patch, label_patch, class_idx

grid = normalize_features_per_city(grid, normalize_features, nodata=NODATA)
print(grid[features].describe())

city_rasters = {}  # city -> (feature_array, label_array, transform, minx, maxy)

for city in grid['city'].unique():
    city_grid = grid[grid['city'] == city].copy()
    feat_arr, label_arr, trs = gdf_to_raster(city_grid, features, 'hwy', cell_width=cell_width)
    minx = city_grid.geometry.centroid.x.min()  # need for coordinate mapping
    maxy = city_grid.geometry.centroid.y.max()
    city_rasters[city] = (feat_arr, label_arr, trs)
    print(f'city {city}: raster shape {feat_arr.shape}')

# rebuild the grid_id -> (city, row, col) map
def gridid_to_rc_map_by_city(grid, cell_width):
    mapping = {}
    for city in grid['city'].unique():
        city_grid = grid[grid['city'] == city]
        centroids = city_grid.geometry.centroid
        xs = centroids.x.values
        ys = centroids.y.values
        minx, miny, maxx, maxy = city_grid.total_bounds
        cols_idx = np.floor((xs - minx) / cell_width).astype(int)
        rows_idx = np.floor((maxy - ys) / cell_width).astype(int)
        for gid, r, c in zip(city_grid['grid_id'].values, rows_idx, cols_idx):
            mapping[int(gid)] = (city, int(r), int(c))
    return mapping

GRIDID_TO_RC = gridid_to_rc_map_by_city(grid, cell_width)

# grid = normalize_features_per_city(grid, normalize_features, nodata=NODATA)
# print(grid[features].describe())
# GRID_FEATURE_ARRAY, GRID_LABEL_ARRAY, rast_transform, CHANNEL_MEANS, CHANNEL_STDS, city_mapping = gdf_to_raster(grid, features, 'hwy', cell_width = 150)
# GRID_LABEL_ARRAY = np.squeeze(GRID_LABEL_ARRAY)
# GRIDID_TO_RC = gridid_to_rc_map(grid, cell_width)

# # filter S_id lists to only include cells with valid label (not NODATA)
# def valid_ids_from_list(id_list):
#     out = []
#     for gid in id_list:
#         if int(gid) in GRIDID_TO_RC:
#             r,c = GRIDID_TO_RC[int(gid)]
#             if 0 <= r < GRID_LABEL_ARRAY.shape[0] and 0 <= c < GRID_LABEL_ARRAY.shape[1]:
#                 if GRID_LABEL_ARRAY[r,c] != NODATA and not np.isnan(GRID_LABEL_ARRAY[r,c]):
#                     out.append(int(gid))
#     return out

# create tensor of the proper size 
batch_tensor = torch.zeros(BATCH_SIZE,nc,2*size_padding+size_potential, 2*size_padding+size_potential) #, dtype=torch.double)
labels = torch.empty(BATCH_SIZE, 2*size_padding+size_potential, 2*size_padding+size_potential, dtype=torch.int64)
# S_id_real = valid_ids_from_list([int(g) for g in hwys])

if isinstance(candidate_list, dict):
    cand_flat = [int(x) for vals in candidate_list.values() for x in (vals or [])]
    S_id_random = cand_flat  # Use the flattened list of grid IDs
elif isinstance(candidate_list, pd.Series):
    cand_flat = [int(x) for vals in candidate_list.tolist() for x in (vals or [])]
    S_id_random = candidate_list['grid_id'].tolist()
else:
    cand_flat = [int(x) for x in candidate_list]
    S_id_random = candidate_list['grid_id'].tolist()

S_id_real = hwys
S_id_real = np.array(S_id_real, dtype=int)
S_id_random = np.array(S_id_random, dtype=int)

# def extract_label_patch(label_array, row, col, window, pad_value = NODATA):
#     pad = window // 2
#     H, W = label_array.shape
#     lbl_p = np.full((H + 2*pad, W + 2*pad), pad_value, dtype=label_array.dtype)
#     lbl_p[pad:pad+H, pad:pad+W] = label_array
#     r0, c0 = row + pad, col + pad
#     return lbl_p[r0-pad:r0+pad+1, c0-pad:c0+pad+1]

def create_batch(batch_tensor=batch_tensor, labels=labels, sample_ids_real=S_id_real, sample_ids_random=S_id_random, return_transf=False):
    batch_tensor = batch_tensor*0
    labels = labels*0

    if return_transf:
        transf = np.zeros(shape=(BATCH_SIZE,5))

    window = 2*size_padding + size_potential
    pad = window // 2

    # guard: ensure sample sets not empty
    if len(sample_ids_real) == 0 and len(sample_ids_random) == 0:
        raise RuntimeError("sample_ids_real or sample_ids_random is empty after filtering for valid labels")

    for b in range(BATCH_SIZE):

        if b < BATCH_SIZE_real + BATCH_SIZE_fill:
            if len(sample_ids_real) == 0:
                s_id = int(np.random.choice(sample_ids_random))
            else:
                s_id = int(np.random.choice(sample_ids_real))
        else:
            if len(sample_ids_random) == 0:
                s_id = int(np.random.choice(sample_ids_real))
            else:
                s_id = int(np.random.choice(sample_ids_random))

        # find raster row/col for this grid_id
        if int(s_id) not in GRIDID_TO_RC:
            continue
        city, row, col = GRIDID_TO_RC[int(s_id)]
        feat_arr, label_arr, trs = city_rasters[city]

        # random augmentations (keep same semantics as original)
        rot_k = np.random.randint(0, 4)
        mirror_var = (np.random.rand() > 0.5)*2 - 1
        shift_x_cells = np.random.randint(-size_potential, size_potential)
        shift_y_cells = np.random.randint(-size_potential, size_potential)

        patch, label_patch, class_idx = extract_patch_from_arrays(feat_arr, label_arr, row, col, window, rot_k, mirror_var, shift_x_cells, shift_y_cells)

        # if patch channels do not match nc, truncate or pad with zeros
        C = patch.shape[0]
        if C != nc:
            print(f"Warning: Patch channels ({C}) do not match expected channels ({nc}). Adjusting...")
        if C >= nc:
            patch = patch[:nc, :, :]
        else:
            pad_ch = np.zeros((nc - C, window, window), dtype='float32')
            patch = np.vstack([patch, pad_ch])

        # verify patch dimensions
        if patch.shape[1:] != (window, window):
            raise ValueError(f"Patch dimensions {patch.shape[1:]} do not match expected size ({window}, {window})")

        # assign patch channels to grid tensor
        for ch in range(nc):
            batch_tensor[b, ch, :, :] = torch.from_numpy(patch[ch])

        # assign label to label tensor
        label_patch_tensor = torch.from_numpy(label_patch)
        clean_label_patch = torch.where((label_patch_tensor == 0) | (label_patch_tensor == 1),
                                        label_patch_tensor.float(),
                                        torch.zeros_like(label_patch_tensor.float()))
        labels[b] = clean_label_patch

        # 'zero' out highways in the 'real' samples
        if b < BATCH_SIZE_real:
            batch_tensor[b, features.index('hwy'), :, :] = 0

    if not return_transf:
        return batch_tensor, labels
    else:
        return batch_tensor, labels, transf

####################################################################################################
### NEURAL NET FUNCTIONS AND INITIALIZATION###
# define neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        main = nn.Sequential(
            # nn.InstanceNorm2d(num_features=nc, affine=True),
            nn.Conv2d(in_channels=nc, out_channels=2*nc, kernel_size=5, padding=2, padding_mode='replicate', bias=True),
            nn.InstanceNorm2d(num_features=2*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2*nc, out_channels=4*nc, kernel_size=11, padding=10, padding_mode='replicate', dilation=2, bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc, out_channels=4*nc, kernel_size=5, padding=2, padding_mode='replicate', bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc, out_channels=1, kernel_size=11, padding=10, padding_mode='replicate', dilation=2, bias=True)
        )
        self.main = main

    def forward(self, x):
        return self.main(x)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Get per-sample cross entropy loss
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape (B,)
        
#         # Compute pt (probability for true class)
#         pt = torch.exp(-ce_loss)  # shape (B,)

#         # Apply focal loss formula
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss

# initialize optimizer
def intitialize_optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# functions to save and load model
def save_model(filename=None):
    if not filename:
        date = (datetime.now(timezone.utc) + timedelta(hours=-7)).strftime('%Y-%m-%d--%H-%M')
        filename = 'checkpoint-epoch-' + str(curr_epoch) + '-' + date + '.tar'
    path_save = outputroot + filename
    # save the model
    torch.save({
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'curr_epoch': curr_epoch,
                'epoch_set_seed': epoch_set_seed,
                }, path_save)
    print('file: ' + path_save)

def load_model(filename, net=None, optimizer=None):
    global epoch_set_seed
    global curr_epoch
    # allow either a path relative to outputroot or an absolute/full path
    path_load = filename if Path(filename).is_absolute() or Path(filename).exists() else Path(outputroot) / filename
    path_load = str(path_load)
    if not net:
        net = Net()
    # if using GPU
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    if not optimizer:
        optimizer = intitialize_optimizer(net)
    if use_cuda and torch.cuda.is_available():
        print('CUDA available')
        checkpoint = torch.load(path_load)
    else:
        print('no CUDA...')
        device = torch.device('cpu')
        checkpoint = torch.load(path_load, map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    curr_epoch = checkpoint.get('curr_epoch', 0)
    if 'epoch_set_seed' in checkpoint.keys():
        epoch_set_seed = checkpoint['epoch_set_seed']
        print('found list in keys')
    epoch_set_seed.append(curr_epoch)
    return net, optimizer

# locations for training and evaluation
num_distinct_train_real = int(len(S_id_real) * frac_train_real)
num_distinct_train_random = int(len(S_id_random) * frac_train_random)

sample_train_real = list(np.random.choice(a=S_id_real,size=num_distinct_train_real,replace=False))
sample_train_real.sort()
sample_train_random = list(np.random.choice(a=S_id_random,size=num_distinct_train_random,replace=False))
sample_train_random.sort()

if num_distinct_train_real < len(S_id_real):
    sample_eval_real = list(set(S_id_real) - set(sample_train_real))
else:
    sample_eval_real = S_id_real
sample_eval_real.sort()
if num_distinct_train_random < len(S_id_random):
    sample_eval_random = list(set(S_id_random) - set(sample_train_random))
else:
    sample_eval_random = S_id_random
sample_eval_random.sort()

# initialize neural net
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def focal_loss(outputs, labels, weights, gamma=2.0):
    ce = F.cross_entropy(outputs, labels, reduction="none")
    # p_t = probability of the true class
    pt = torch.exp(-ce)
    # Focal term
    focal_term = (1 - pt) ** gamma
    # Class weights applied per sample
    class_w = weights[labels]
    # Weighted focal loss
    loss = (class_w * focal_term * ce).mean()
    return loss

def compute_batch_weights(labels, num_classes):
    # count frequencies in the batch
    valid_mask = (labels >= 0) & (labels < num_classes)
    valid_labels = labels[valid_mask]
    unique, counts = torch.unique(valid_labels, return_counts=True)
    unique = unique.long()
    freq = torch.zeros(num_classes, device=labels.device)
    freq[unique] = counts.float()

    # avoid division by zero (classes not in batch)
    freq = torch.where(freq > 0, freq, torch.tensor(1.0, device=labels.device))

    # sqrt-inverse frequency
    weights = 1.0 / torch.sqrt(freq)
    weights[0] *= 2.0

    # normalize mean weight = 1
    weights = weights / weights.mean()

    return weights

if use_saved_model:
    print('Loading model')
    net, optimizer = load_model(saved_model_filename)
else:
    net = Net()
    net.apply(weights_init)
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    optimizer = intitialize_optimizer(net)

# # Ensure GRID_LABEL_ARRAY contains only binary labels (0 and 1)
# GRID_LABEL_ARRAY = np.where((GRID_LABEL_ARRAY == 0) | (GRID_LABEL_ARRAY == 1), GRID_LABEL_ARRAY, 0)

# # Compute class weights for binary classification
# unique, counts = np.unique(GRID_LABEL_ARRAY, return_counts=True)
# if len(counts) != 2:
#     raise ValueError(f"Expected 2 classes, but found {len(counts)} unique labels: {unique}")
# total = counts.sum()
# class_weights = torch.tensor([total / (2.0 * c) for c in counts], dtype=torch.float32).to('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

# # Define the weighted loss function
# # criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = FocalLoss(alpha = 0.25, gamma = 2.0)
# if use_cuda and torch.cuda.is_available():
#     print('using CUDA')
#     net.cuda()

# Set random seed for reproducibility: increment to ensure different training samples after load
manualSeed = 24601 + curr_epoch
#manualSeed = random.randint(1, 10000) # use if you want new results
print('Random Seed: ', manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

# transform output into list
def reverse_augmentation_to_coordinates(coords, theta_deg=0.0, mirror_var=1, shift_x_pixels=0.0, shift_y_pixels=0.0):
    theta_rad = -theta_deg * np.pi / 180  # Reverse rotation
    if mirror_var == -1:
        coords[:, 0] = -coords[:, 0]  # Reverse mirroring
    if theta_deg != 0.0:
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                               [np.sin(theta_rad), np.cos(theta_rad)]])
        coords = coords @ rot_matrix.T
    coords[:, 0] -= shift_x_pixels  # Reverse X shift
    coords[:, 1] -= shift_y_pixels  # Reverse Y shift
    return coords

def add_to_list(s_id, o, r, city):
    list_out.append((s_id, o.tolist(), r, city))


def outputs_to_loc(outputs,transf):
    o = torch.softmax(outputs, dim=1).cpu().numpy()
    
    for b in range(BATCH_SIZE):
        s_id = int(transf[b, 0])
        add_to_list(s_id, o[b], b < BATCH_SIZE_real, city)

# run prediction loop
list_out = list()

with torch.no_grad():
    for i in range(num_batches_predict):
        # show progress
        if i % 100 == 0:
            print(str(i+1) + "/" + str(num_batches_predict) + " - time: " + (datetime.now(timezone.utc) + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))

        # get the inputs; data is a list of [inputs, labels]
        data = create_batch(return_transf=True)
        inputs, labels, transf = data

        if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        outputs_to_loc(outputs,transf)

# print(list_out)

print(len(list_out))

# save resulting file
date = (datetime.now(timezone.utc) + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S')
filename_out = 'predicted_activation-' + date + '.csv'

with open(dataroot+filename_out,'w') as f:
    f.write('s_id,real_missing,prob_1,prob_0,city\n')  
    for e in list_out:
        s_id = str(e[0])  
        real_missing = str(e[2])  
        prob_1 = e[1][0]  
        prob_0 = e[1][1]  
        f.write(f"{s_id},{real_missing},{prob_1},{prob_0}\n")  

# cleanup function - only keep the 3 most recent csvs
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))

# Sort files by modification time (most recent first)
csv_files.sort(key=os.path.getmtime, reverse=True)

# Keep only the most recent 3 files
files_to_keep = csv_files[:3]
files_to_delete = csv_files[3:]

# Delete older files
for file in files_to_delete:
    os.remove(file)
    print(f"Deleted: {file}")

print(f"Kept the most recent 3 files: {files_to_keep}")