import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_sample(path='data/output/sample.pkl'):
    """Load the grid-square sample and construct the variables used across specifications."""
    df = pd.read_pickle(path)
    df['rent'] = df['rent'].replace(0, 0.00001)
    df['valueh'] = df['valueh'].replace(0, 0.00001)

    df['log_valueh'] = np.log(df['valueh']) * df['valueh_avail']
    df['log_rent'] = np.log(df['rent']) * df['rent_avail']
    df['city_louisville'] = (df['city'] == 'louisville').astype(int)
    df['city_littlerock'] = (df['city'] == 'littlerock').astype(int)
    df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
    df['log_dist_to_rr'] = np.log(df['dist_to_rr'])
    df['log_dist_to_rr_sq'] = df['log_dist_to_rr'] ** 2
    df['log_dist_to_hwy'] = np.log(df['dist_to_hwy'])
    df['ResidentialxBlack'] = df['Residential'] * df['mblack_1945def']
    return df


def restrict_to_discretionary(df):
    """Restrict to grid squares that are not part of the 1940 highway network and not
    adjacent to it, i.e. squares where placement in later decades was discretionary."""
    hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].copy()
    all_squares = df[['grid_id', 'geometry']].copy()
    touches_result = gpd.sjoin(
        all_squares,
        hwy_40_squares[['geometry']],
        how='left',
        predicate='touches'
    )
    adjacent_ids = set(
        touches_result[touches_result['index_right'].notna()]['grid_id']
    )
    return df[~df['grid_id'].isin(adjacent_ids) & (df['hwy_40'] == 0)].copy()


def merge_cnn_probs(df, model_pattern, dataroot='cnn/'):
    """Merge in predicted P(highway) from the most recently modified CNN output file
    matching model_pattern (e.g. 'predicted_activation-model1*.csv')."""
    matches = sorted(
        glob.glob(os.path.join(dataroot, model_pattern)),
        key=os.path.getmtime,
        reverse=True,
    )
    logits_df = pd.read_csv(matches[0])
    logits_df['grid_id'] = logits_df['grid_id'].astype(str)

    orig_dtype = df['grid_id'].dtype
    df = df.copy()
    df['grid_id'] = df['grid_id'].astype(str)
    df = df.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
    df['grid_id'] = df['grid_id'].astype(orig_dtype)
    return df


def add_cnn_interactions(df):
    """Add interaction terms between the CNN-predicted highway probability and
    Residential/Black, used in specifications that condition on the CNN covariate."""
    df = df.copy()
    df['BlackxProbHwy'] = df['mblack_1945def'] * df['prob_hwy']
    df['ResidentialxProbHwy'] = df['Residential'] * df['prob_hwy']
    df['ResidentialxBlackxProbHwy'] = df['Residential'] * df['mblack_1945def'] * df['prob_hwy']
    return df


def split_by_candidates(df, candidate_dict):
    """Split a sample into the 'direct' subset (grid squares that were ML/manual
    candidates for highway placement in their city) and the 'indirect' complement."""
    direct_frames, indirect_frames = [], []
    for city in df['city'].unique():
        candidates = candidate_dict[city]
        city_df = df.loc[df['city'] == city]
        is_candidate = city_df['grid_id'].isin(candidates)
        direct_frames.append(city_df.loc[is_candidate].copy())
        indirect_frames.append(city_df.loc[~is_candidate].copy())
    direct = pd.concat(direct_frames, ignore_index=True)
    indirect = pd.concat(indirect_frames, ignore_index=True)
    return direct, indirect

def compute_demographic_access(grid, demographic_var, decay_m, max_dist_m = 5000):
    centroids = grid.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    dists = cdist(coords, coords, metric = 'euclidean')

    # compute distance decay weights
    weights = np.exp(-dists / decay_m)
    weights[dists>max_dist_m] = 0
    np.fill_diagonal(weights, 0)

    demo_vals = grid[demographic_var].fillna(0).values

    # weighted sum and normalized 
    access = weights @ demo_vals
    weight_sums = weights.sum(axis=1)
    dem_access_norm = np.where(weight_sums > 0, access / weight_sums, 0)

    grid = grid.copy()
    grid['dem_access_norm'] = dem_access_norm
    grid['dem_access_raw'] = access
    grid['ResidentialxAccess'] = grid['Residential'] * grid['dem_access_norm']
    return grid