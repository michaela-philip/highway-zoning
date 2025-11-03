from datetime import datetime, timedelta
import pandas as pd
import geopandas as gpd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import sin, cos, pi  # rotating regions
from math import floor  # truncating naics codes
import rasterio
from rasterio import transform
from pathlib import Path
from scipy.ndimage import rotate, shift

# Root directory for dataset
dataroot = 'cnn/'
outputroot = 'cnn/'

saved_model_filename = 'checkpoint-epoch-<epoch>-YYYY-MM-DD--hh-mm.tar'

####################################################################################################
### PARAMETERS ###

# read in data and prepare lists
sample = pd.read_pickle('data/input/samplelist.pkl')
candidate_list = pd.read_pickle('data/output/cnn_candidate_list.pkl')
grid = pd.read_pickle('data/output/sample.pkl')
hwys = grid[grid['hwy'] == 1]['grid_id'].unique().tolist()
features = ['valueh', 'rent', 'distance_to_cbd', 'dist_water', 'owner', 'dm_elevation']

cell_width = 150  # cell width in meters (convert from miles)
size_potential = 6  # potential locations: num_width_potential x num_width_potential
size_padding = 5  # number of padding cells on each side of potential grid
nc = len(features)  # number of channels: 1) other grocery stores 2) other businesses

num_batches_predict = 5000
# change ratio of what regions to simulate since we don't care much about the real ones
BATCH_SIZE_real = 2  # regions with missing grocery store per batch
BATCH_SIZE_fill = 2  # regions with real location filled in (-> no missing) per batch
BATCH_SIZE_random = 28  # random regions (-> no missing) per batch
BATCH_SIZE = BATCH_SIZE_real + BATCH_SIZE_fill + BATCH_SIZE_random

use_cuda = True

curr_epoch = 0
epoch_set_seed = list()
epoch_set_seed.append(curr_epoch)
EPOCHS = 20
ITERS = 5000
NODATA = -9999.0

print('BATCH_SIZE: ' + str(BATCH_SIZE))
print('cell_width: ' + str(round(cell_width)) + 'm')


####################################################################################################
### SAMPLE/BATCH CREATION FUNCTIONS ###
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

    if 'owner' in gdf.columns and gdf['owner'].dtype == object:
        owners = pd.factorize(gdf['owner'])[0]  # integers 0..N-1
        gdf = gdf.copy()
        gdf['owner'] = owners.astype(float)

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
    label_array = out_array[len(features):, :, :]
    # if label was one band, squeeze to 2D
    if label_array.shape[0] == 1:
        label_array = np.squeeze(label_array, axis=0)

    # compute numeric means/stds (fallback to 0.0/1.0)
    means, stds = [], []
    for i in range(len(features)):
        band = out_array[i]
        mask = band != nodata
        if mask.any():
            mu = float(np.mean(band[mask]))
            sd = float(np.std(band[mask]))
            if sd == 0 or np.isnan(sd):
                sd = 1.0
        else:
            mu, sd = 0.0, 1.0
        means.append(mu)
        stds.append(sd)

    return feature_array, label_array, trs, means, stds

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
def extract_patch_from_arrays(feature_array, row, col, window, pad_value = 0.0):
    C, H, W = feature_array.shape
    pad = window // 2
    arr_p = np.pad(feature_array, ((0, 0), (pad, pad), (pad, pad)), mode = 'constant', constant_values = pad_value)
    r0, c0 = row + pad, col + pad
    patch = arr_p[:, r0-pad:r0+pad, c0-pad:c0+pad]
    return patch.astype('float32')

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

GRID_FEATURE_ARRAY, GRID_LABEL_ARRAY, rast_transform, CHANNEL_MEANS, CHANNEL_STDS = gdf_to_raster(grid, features, 'hwy', cell_width = 150)
GRID_LABEL_ARRAY = np.squeeze(GRID_LABEL_ARRAY)
GRIDID_TO_RC = gridid_to_rc_map(grid, cell_width)

# filter S_id lists to only include cells with valid label (not NODATA)
def valid_ids_from_list(id_list):
    out = []
    for gid in id_list:
        if int(gid) in GRIDID_TO_RC:
            r,c = GRIDID_TO_RC[int(gid)]
            if 0 <= r < GRID_LABEL_ARRAY.shape[0] and 0 <= c < GRID_LABEL_ARRAY.shape[1]:
                if GRID_LABEL_ARRAY[r,c] != NODATA and not np.isnan(GRID_LABEL_ARRAY[r,c]):
                    out.append(int(gid))
    return out

# create tensor of the proper size 
batch_tensor = torch.zeros(BATCH_SIZE,nc,2*size_padding+size_potential, 2*size_padding+size_potential) #, dtype=torch.double)
labels = torch.empty(BATCH_SIZE, dtype=torch.int64)
S_id_real = valid_ids_from_list([int(g) for g in hwys])

if isinstance(candidate_list, dict):
    cand_flat = [int(x) for vals in candidate_list.values() for x in (vals or [])]
elif isinstance(candidate_list, pd.Series):
    cand_flat = [int(x) for vals in candidate_list.tolist() for x in (vals or [])]
else:
    cand_flat = [int(x) for x in candidate_list]

cand_flat = cand_flat  # keep existing construction above
S_id_random = valid_ids_from_list([g for g in cand_flat if g not in S_id_real])

S_id_real = np.array(S_id_real, dtype=int)
S_id_random = np.array(S_id_random, dtype=int)

def extract_label_patch(label_array, row, col, window, pad_value = NODATA):
    pad = window // 2
    H, W = label_array.shape
    lbl_p = np.full((H + 2*pad, W + 2*pad), pad_value, dtype=label_array.dtype)
    lbl_p[pad:pad+H, pad:pad+W] = label_array
    r0, c0 = row + pad, col + pad
    return lbl_p[r0-pad:r0+pad+1, c0-pad:c0+pad+1]

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
        
         # extract label patch, apply same augmentation with nearest interpolation, and read center
        lbl_patch = extract_label_patch(GRID_LABEL_ARRAY, row, col, window, pad_value=NODATA)
        lbl_patch_ch = lbl_patch[np.newaxis, :, :]  # shape (1,H,W)
        lbl_aug = apply_augmentation_to_patch(lbl_patch_ch,
                                             theta_deg=theta,
                                             mirror_var=mirror_var,
                                             shift_x_pixels=shift_x_pixels,
                                             shift_y_pixels=shift_y_pixels,
                                             order=0,  # nearest for labels
                                             cval=NODATA)
        center_label = lbl_aug[0, pad, pad]
        # handle nodata after augmentation (rare) — fallback to original center if available or set 0
        if center_label == NODATA or np.isnan(center_label):
            original_center = GRID_LABEL_ARRAY[row, col]
            if original_center == NODATA or np.isnan(original_center):
                center_label = 0
            else:
                center_label = int(original_center)
        labels[b] = int(center_label)

    # normalize each channel using precomputed means/stds (unchanged)
    for ch in range(patch.shape[0]):
        mu = CHANNEL_MEANS[ch] if ch < len(CHANNEL_MEANS) else 0.0
        sd = CHANNEL_STDS[ch] if ch < len(CHANNEL_STDS) else 1.0
        if sd == 0 or np.isnan(sd):
            sd = 1.0
        patch[ch] = (patch[ch] - mu) / sd

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

    # if you want to include the center cell as in original 'filled' case:
    if b >= BATCH_SIZE_real and b < BATCH_SIZE_real + BATCH_SIZE_fill:
        treat_x = pad
        treat_y = pad
        batch_tensor[b, 0, treat_y, treat_x] += 1


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
            # binary classification => 2 logits
            nn.Linear(1 * pow(2*size_padding+size_potential, 2), 2),
        )
        self.main = main

    def forward(self, x):
        return self.main(x)

# initialize optimizer
def intitialize_optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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

# fine-tuning function 
def fine_tune_on_city(sample_train_real_city, sample_train_random_city,
                      base_model_path='cnn/base_pooled_model.tar',
                      fine_epochs=5, fine_iters=2000, lr=1e-3, freeze_backbone=True):
    # load base pooled model
    net_ft, optimizer_ft = load_model(base_model_path)
    # freeze backbone if requested (unfreeze classifier = last Linear)
    if freeze_backbone:
        for p in net_ft.parameters():
            p.requires_grad = False
        # find last linear module (assumes net.main ends with Linear)
        if isinstance(net_ft.main[-1], nn.Linear):
            for p in net_ft.main[-1].parameters():
                p.requires_grad = True
    # optimizer only for trainable params
    trainable = [p for p in net_ft.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found; check freeze_backbone setting")
    optimizer_city = optim.SGD(trainable, lr=lr, momentum=0.9)

    # compute class weights from available city labels (optional)
    def get_labels_for_ids(id_list):
        labs = []
        for gid in id_list:
            if int(gid) in GRIDID_TO_RC:
                r,c = GRIDID_TO_RC[int(gid)]
                labs.append(int(GRID_LABEL_ARRAY[r, c]))
        return np.array(labs, dtype=int)

    y_real = get_labels_for_ids(sample_train_real_city)
    y_rand = get_labels_for_ids(sample_train_random_city)
    y_all = np.concatenate([y_real, y_rand]) if len(y_real) + len(y_rand) > 0 else np.array([0,1])
    vals, counts = np.unique(y_all, return_counts=True)
    total = counts.sum()
    weights = np.ones(2, dtype=np.float32)
    for v, c in zip(vals, counts):
        weights[int(v)] = float(total) / (2.0 * float(c))
    criterion_city = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(next(net_ft.parameters()).device))

    # small training loop using create_batch (pass city-specific sample lists)
    device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')
    net_ft.to(device)
    net_ft.train()
    for epoch in range(fine_epochs):
        running_loss = 0.0
        for it in range(fine_iters):
            data = create_batch(sample_ids_real=sample_train_real_city, sample_ids_random=sample_train_random_city)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_city.zero_grad()
            outputs = net_ft(inputs)
            loss = criterion_city(outputs, labels)
            loss.backward()
            optimizer_city.step()

            running_loss += float(loss.item())
            if (it+1) % max(1, fine_iters//5) == 0:
                avg = running_loss / max(1, fine_iters//5)
                print(f'Fine-tune epoch {epoch+1}/{fine_epochs} iter {it+1}/{fine_iters} loss={avg:.4f}')
                running_loss = 0.0

    # return fine-tuned model and optimizer (caller can save)
    return net_ft, optimizer_city

"""Set random seed"""

# Set random seed for reproducibility
manualSeed = 24601
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""load neural net"""

net, optimizer = load_model(saved_model_filename)

criterion = nn.CrossEntropyLoss()

if use_cuda and torch.cuda.is_available():
    net.cuda()

# Set random seed for reproducibility: increment to ensure different training samples after load
manualSeed = 24601 + curr_epoch
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""Functions to transform output into list for saving to file"""

@numba.jit
def data_reverse_shift_rotate(xy,shift_x=0,shift_y=0,theta=0,mirror_var=1):
    # reverse shift
    xy[:,0] -= shift_x
    xy[:,1] -= shift_y

    # reverse mirroring
    if mirror_var == 1 or mirror_var == -1:
        xy[:,0] = mirror_var * xy[:,0]
    
    # reverse rotation by theta
    theta = theta * pi / 180
    if not theta == 0:
        # x = cos(theta) * mat[:,0] - sin(theta) * mat[:,1]
        # y = sin(theta) * mat[:,0] + cos(theta) * mat[:,1]
        rot = np.array([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
        xy = xy@np.linalg.inv(rot)
        # x = xy[:,0]
        # y = xy[:,1]
    return xy


def add_to_list(xy,o,r):
    for i in range(len(xy)):
        if i == len(xy) - 1:
            tup = (str(int(round(xy[i,0]))), str(r), 'NA', 'NA', str(o[i]))
        else:
            tup = (str(int(round(xy[i,0]))), str(r), str(int(round(xy[i,1]))), str(int(round(xy[i,2]))), str(o[i]))
        list_out.append(tup)


def outputs_to_loc(outputs,transf):
    o = outputs.cpu().numpy()
    g = np.linspace(start=cell_width/2,
                    stop=cell_width/2 + cell_width*size_potential,
                    num=size_potential, endpoint=False)
    
    for b in range(BATCH_SIZE):
        # grid cell midpoints
        xy = np.zeros(shape=(pow(size_potential,2)+1,3))
        # set s_id
        xy[:,0] = int(transf[b,0])
        # set relative location
        xy[0:pow(size_potential,2),1] = np.tile(g,size_potential)
        xy[0:pow(size_potential,2),2] = np.repeat(g,size_potential)

        xy[0:pow(size_potential,2),1:3] = data_reverse_shift_rotate(xy[0:pow(size_potential,2),1:3],
                                                                    shift_x=transf[b,1],
                                                                    shift_y=transf[b,2],
                                                                    theta=transf[b,3],
                                                                    mirror_var=transf[b,4])
        add_to_list(xy,o[b,:],b < BATCH_SIZE_real)

"""run the neural net on many practice examples to guess real locations"""

list_out = list()

with torch.no_grad():
    for i in range(num_batches_predict):
        # show progress
        if i % 100 == 0:
            print(str(i+1) + "/" + str(num_batches_predict) + " - time: " + (datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d %H:%M:%S'))

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

"""Save the resulting file

"""

date = (datetime.utcnow() + timedelta(hours=-7)).strftime('%Y-%m-%d--%H-%M')
filename_out = 'predicted_activation-' + date + '.csv'

with open(dataroot+filename_out,'w') as f:
    f.write('s_id,real_missing,x,y,activation\n')
    for e in list_out:
        f.write(','.join(e) + '\n')