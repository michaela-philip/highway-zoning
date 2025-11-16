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
import glob 
import os


dataroot = 'cnn/multiclass/'
outputroot = 'cnn/multiclass/'

use_saved_model = True
saved_model_filename = 'cnn/multiclass/mc_model2.tar'  # path to saved model for prediction

####################################################################################################
### PARAMETERS ###

# read in data and prepare lists
sample = pd.read_pickle('data/input/samplelist.pkl')
candidate_list = pd.read_pickle('data/output/cnn_candidate_list.pkl')
grid = pd.read_pickle('data/output/sample.pkl')
hwys = grid[grid['hwy'] == 1]['grid_id'].unique().tolist()
features = ['distance_to_cbd', 'dist_water', 'dist_to_hwy', 'elevation', 'city']
normalize_features = ['distance_to_cbd', 'dist_water', 'dist_to_hwy', 'elevation'] # the only features i want to demean
obs = grid.shape[0]

cell_width = 150  # cell width in meters (convert from miles)
size_potential = 4  # potential locations: num_width_potential x num_width_potential
size_padding = 2  # number of padding cells on each side of potential grid
nc = len(features) + 2  # number of channels (2 additional for coordinates)
BATCH_SIZE_real = 2  # regions with hwy (but removed)
BATCH_SIZE_fill = 2  # regions with hwy (not removed)
BATCH_SIZE_random = 28  # regions with no hwy (but candidates)
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
        cities = pd.factorize(gdf['city'])[0]  # integers 0..N-1
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

    outpath = Path('data/output/rasterized.tif')
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(outpath), 'w', **profile) as dst:
        for i in range(nb):
            dst.write(out_array[i], i+1)
    print(f'rasterization complete, file saved: {outpath}')

    feature_array = out_array[:len(features), :, :]

    return feature_array, trs

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
# def extract_patch_from_arrays(feature_array, row, col, window, pad_value=0.0):
#     C, H, W = feature_array.shape
#     pad = window // 2

#     # Create an empty patch with padding value
#     patch = np.full((C, window, window), pad_value, dtype=feature_array.dtype)

#     # Compute the bounds of the patch
#     r_start = max(0, row - pad)
#     r_end = min(H, row + pad + 1)
#     c_start = max(0, col - pad)
#     c_end = min(W, col + pad + 1)

#     # Compute the corresponding indices in the patch
#     pr_start = max(0, pad - row)
#     pr_end = pr_start + (r_end - r_start)
#     pc_start = max(0, pad - col)
#     pc_end = pc_start + (c_end - c_start)

#     # Copy the valid region from the feature array to the patch
#     patch[:, pr_start:pr_end, pc_start:pc_end] = feature_array[:, r_start:r_end, c_start:c_end]

#     return patch.astype('float32')
def extract_patch_from_arrays(feature_array, row, col, window, pad_value = 0.0):
    C, H, W = feature_array.shape
    pad = window // 2

    # Initialize patch with zeros
    patch = np.zeros((C, window, window), dtype=np.float32)

    # Compute source indices
    r_start = max(0, row - pad)
    r_end = min(H, row + pad + 1)
    c_start = max(0, col - pad)
    c_end = min(W, col + pad + 1)

    # Compute destination indices in patch
    pr_start = max(0, pad - row)
    pr_end = pr_start + (r_end - r_start)
    pc_start = max(0, pad - col)
    pc_end = pc_start + (c_end - c_start)

    # Copy features into patch
    patch[:, pr_start:pr_end, pc_start:pc_end] = feature_array[:, r_start:r_end, c_start:c_end]

    # Add coordinate channels
    x_coords = np.linspace(-1, 1, window, dtype=np.float32)
    y_coords = np.linspace(-1, 1, window, dtype=np.float32)
    x_channel = np.tile(x_coords, (window, 1))          # shape (window, window)
    y_channel = np.tile(y_coords[:, np.newaxis], (1, window))

    patch_with_coords = np.concatenate([
        patch,
        x_channel[np.newaxis, :, :],
        y_channel[np.newaxis, :, :]
    ], axis=0)

    return patch_with_coords.astype('float32')

def apply_augmentation_to_patch(patch, theta_deg=0.0, mirror_var=1, shift_x_pixels=0.0, shift_y_pixels=0.0, order=1, cval=0.0):
    C, H, W = patch.shape
    out = np.empty_like(patch)
    for ch in range(C):
        layer = patch[ch]
        # mirror left-right if requested
        if mirror_var == -1:
            layer = np.fliplr(layer)
        # rotate about center (reshape=False keeps same H,W)
        if theta_deg != 0.0:
            layer = rotate(layer, angle=theta_deg, reshape=False, order=order, mode='constant', cval=cval)
        # shift (rows, cols)
        if shift_y_pixels != 0.0 or shift_x_pixels != 0.0:
            layer = shift(layer, shift=(shift_y_pixels, shift_x_pixels), order=order, mode='constant', cval=cval)
        out[ch] = layer
    return out

grid = normalize_features_per_city(grid, normalize_features, nodata=NODATA)
print(grid[features].describe())
GRID_FEATURE_ARRAY, rast_transform = gdf_to_raster(grid, features, 'hwy', cell_width = 150)
GRIDID_TO_RC = gridid_to_rc_map(grid, cell_width)

# create tensor of the proper size 
batch_tensor = torch.zeros(BATCH_SIZE,nc,2*size_padding+size_potential + 1, 2*size_padding+size_potential + 1) #, dtype=torch.double)
labels = torch.empty(BATCH_SIZE, dtype=torch.int64)
S_id_real = hwys

if isinstance(candidate_list, dict):
    cand_flat = [int(x) for vals in candidate_list.values() for x in (vals or [])]
    S_id_random = cand_flat  # Use the flattened list of grid IDs
elif isinstance(candidate_list, pd.Series):
    cand_flat = [int(x) for vals in candidate_list.tolist() for x in (vals or [])]
    S_id_random = candidate_list['grid_id'].tolist()
else:
    cand_flat = [int(x) for x in candidate_list]
    S_id_random = candidate_list['grid_id'].tolist()

S_id_real = np.array(S_id_real, dtype=int)
S_id_random = np.array(S_id_random, dtype=int)

def create_batch(batch_tensor=batch_tensor, labels=labels, sample_ids_real=S_id_real, sample_ids_random=S_id_random, return_transf=False):
    batch_tensor = batch_tensor*0
    labels = labels*0

    if return_transf:
        transf = np.zeros(shape=(BATCH_SIZE,5))

    window = 2*size_padding + size_potential + 1
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
            # if grid_id not found, pick arbitrary valid center
            H = GRID_FEATURE_ARRAY.shape[1]; W = GRID_FEATURE_ARRAY.shape[2]
            row = np.random.randint(pad, H-pad)
            col = np.random.randint(pad, W-pad)
        else:
            row, col = GRIDID_TO_RC[int(s_id)]

        # random augmentations (keep same semantics as original)
        theta = np.random.rand()*360
        mirror_var = (np.random.rand() > 0.5)*2 - 1
        shift_x = np.random.rand()*cell_width*size_potential - cell_width/2
        shift_y = np.random.rand()*cell_width*size_potential - cell_width/2

        if return_transf:
            transf[b,0] = s_id
            transf[b,1] = shift_x
            transf[b,2] = shift_y
            transf[b,3] = theta
            transf[b,4] = mirror_var

        patch = extract_patch_from_arrays(GRID_FEATURE_ARRAY, row, col, window, pad_value=0.0)

        # convert shifts in meters to pixels for ndimage.shift
        shift_x_pixels = shift_x / float(cell_width)   # positive -> move right
        shift_y_pixels = shift_y / float(cell_width)   # positive -> move down

        # apply geometric augmentation (mirror, rotate, shift) to all channels
        # use bilinear interpolation (order=1) for continuous channels
        patch = apply_augmentation_to_patch(patch,
                                            theta_deg=theta,
                                            mirror_var=mirror_var,
                                            shift_x_pixels=shift_x_pixels,
                                            shift_y_pixels=shift_y_pixels,
                                            order=1,
                                            cval=0.0)

        # true location of 'missing' grocery store
        if b >= BATCH_SIZE_real and b < BATCH_SIZE_real+BATCH_SIZE_fill:
            treat_x = int(round(shift_x_pixels/cell_width) + size_padding)
            treat_y = int(round(shift_y_pixels/cell_width) + size_padding)
            batch_tensor[b,0,treat_y,treat_x] += 1

        # label with location of 'missing' grocery store:
        if b < BATCH_SIZE_real:
            labels[b] = int(round(shift_y_pixels)*size_potential) + int(round(shift_x_pixels))
        # random region without missing grocery store or grocery store is filled in:
        else:
            labels[b] = pow(size_potential,2)  # index 1 larger than locations (start at 0)

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
            nn.InstanceNorm2d(num_features=nc, affine=True),
            nn.Conv2d(in_channels=nc, out_channels=2*nc, kernel_size=5, padding=2, padding_mode='replicate', bias=True),
            nn.InstanceNorm2d(num_features=2*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2*nc, out_channels=4*nc, kernel_size=21, padding=20, padding_mode='replicate', dilation=2, bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc, out_channels=4*nc, kernel_size=5, padding=2, padding_mode='replicate', bias=True),
            nn.InstanceNorm2d(num_features=4*nc, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4*nc, out_channels=1, kernel_size=21, padding=20, padding_mode='replicate', dilation=2, bias=True),
            nn.InstanceNorm2d(num_features=1, affine=True),
            nn.Flatten(),
            nn.Linear(1*pow(2*size_padding+size_potential + 1,2), pow(size_potential,2)+1),
        )
        self.main = main

    def forward(self, x):
        return self.main(x)

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

# separate locations for training and evaluation
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
if use_saved_model:
    print('Loading model')
    net, optimizer = load_model(saved_model_filename)
else:
    net = Net()
    net.apply(weights_init)
    if use_cuda and torch.cuda.is_available():
        net.cuda()
    optimizer = intitialize_optimizer(net)

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
    unique, counts = torch.unique(labels, return_counts=True)
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

# criterion = nn.CrossEntropyLoss()

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

def add_to_list(xy, o, r):
    for i in range(len(xy)):
        if i == len(xy) - 1:
            tup = (str(int(round(xy[i,0]))), str(r), 'NA', 'NA', str(o[i]))
        else:
            tup = (str(int(round(xy[i,0]))), str(r), str(int(round(xy[i,1]))), str(int(round(xy[i,2]))), str(o[i]))
        list_out.append(tup)


def outputs_to_loc_multiclass(outputs, transf, size_potential, cell_width):
    o = outputs.cpu().numpy()  # [B, N+1]
    # BATCH_SIZE = o.shape[0]
    N = size_potential ** 2

    # Compute midpoints of each grid cell in patch
    g = np.linspace(start=cell_width/2,
                    stop=cell_width/2 + cell_width*size_potential,
                    num=size_potential, endpoint=False)

    for b in range(BATCH_SIZE):
        # Initialize xy array: [num_cells + 1, 3] -> s_id, x, y
        xy = np.zeros((N + 1, 3))
        xy[:,0] = int(transf[b,0])  # s_id
        xy[:N,1] = np.tile(g, size_potential)
        xy[:N,2] = np.repeat(g, size_potential)

        # Reverse augmentation on grid coordinates
        xy[:N,1:3] = reverse_augmentation_to_coordinates(
            xy[:N,1:3],
            theta_deg=transf[b,3],
            mirror_var=transf[b,4],
            shift_x_pixels=transf[b,1]/float(cell_width),
            shift_y_pixels=transf[b,2]/float(cell_width)
        )

        # Append to list with activation values
        add_to_list(xy, o[b,:], b < BATCH_SIZE_real)

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

        outputs_to_loc_multiclass(outputs,transf, size_potential, cell_width)


print(len(list_out))

# save resulting file
date = (datetime.now(timezone.utc) + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S')
filename_out = 'predicted_activation-' + date + '.csv'

with open(dataroot+filename_out,'w') as f:
    f.write('s_id,real_missing,x,y,activation\n')
    for e in list_out:
        f.write(','.join(e) + '\n')

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