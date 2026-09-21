import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import shapely.geometry

def load_sample(size, impute = False):
    """Load the grid-square sample and construct the variables used across specifications."""
    path=f'data/output/sample_{size}.pkl'
    df = pd.read_pickle(path)
    if impute:
        df = impute_values(df, columns=df.columns)
    else:
        df = df[df['imputed'] == 0].copy()
    # df['rent'] = df['rent'].replace(0, 0.00001)
    # df['valueh'] = df['valueh'].replace(0, 0.00001)
    # df = df[df['imputed'] == 0].copy()
    df['log_valueh'] = np.log(df['valueh'] + 0.00001) * df['valueh_avail']
    df['log_rent'] = np.log(df['rent'] + 0.00001) * df['rent_avail']
    df['city_louisville'] = (df['city'] == 'louisville').astype(int)
    df['city_littlerock'] = (df['city'] == 'littlerock').astype(int)
    df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
    df['log_dist_to_rr'] = np.log(df['dist_to_rr'])
    df['log_dist_to_rr_sq'] = df['log_dist_to_rr'] ** 2
    df['log_dist_to_hwy'] = np.log(df['dist_to_hwy'])

    df['slope'] = 100 * df['slope']  # convert to percent slope
    df['mean_hwy_construction'] = df.groupby('city')['hwy'].transform('mean')
    df['pct_black_owner'] = np.where(df['owner'] != 0, df['black_homeowners'] / (df['owner'] * df['serial']), 0)
    df['black_homeowners_indicator'] = np.where(df['black_homeowners'] > 0, 1, 0)
    df['homeowners'] = df['owner'] * df['serial']

    df['mblack_1945def'] = np.where(df['pct_black'] >= 0.6, 1, 0)
    df['mblack_50_pct'] = np.where(df['pct_black'] >= 0.5, 1, 0)
    df['mblack_40_pct'] = np.where(df['pct_black'] >= 0.4, 1, 0)
    df['mixed_black'] = np.where((df['pct_black'] >= 0.3) & (df['pct_black'] < 0.8), 1, 0)
    df['mblack_mean_pct'] = np.where(df['pct_black'] >= df['pct_black'].mean(), 1, 0)
    df['mblack_mean_share'] = np.where(df['share_black'] >= df['share_black'].mean(), 1, 0)
    df['log_black'] = np.log(df['pct_black'] + 0.000001)
    df['any_black'] = np.where(df['black_pop'] != 0, 1, 0)
    return df


def add_interaction(df, var_a, var_b, col=None):
    """Return the name of the df column holding var_a * var_b, computing and caching it
    on df (in place) if it isn't already present. Lets spec-building code (see
    analysis.lib.specs.core_spec/sweep_interactions_spec) derive whatever interaction it
    needs from a chosen base variable instead of every combination being precomputed."""
    col = col or f'{var_a}x{var_b}'
    if col not in df.columns:
        df[col] = df[var_a] * df[var_b]
    return col


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
    df = df.merge(logits_df[['grid_id', 'logit_hwy', 'prob_hwy']], on='grid_id', how='left')
    for city in df['city'].unique():
        city_mean = df.loc[df['city'] == city, 'logit_hwy'].mean()
        city_std = df.loc[df['city'] == city, 'logit_hwy'].std()
        df.loc[df['city'] == city, 'logit_centered'] = df.loc[df['city'] == city, 'logit_hwy'] - city_mean
        df.loc[df['city'] == city, 'logit_normalized'] = (df.loc[df['city'] == city, 'logit_hwy'] - city_mean) / city_std
    df['grid_id'] = df['grid_id'].astype(orig_dtype)
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

def compute_characteristic_access(grid, characteristic_var, access_name, decay_m, rho = None, max_dist_m = 5000):
    centroids = grid.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    dists = cdist(coords, coords, metric = 'euclidean')

    # compute distance decay weights
    # weights = np.exp(-dists / decay_m)
    if rho is not None:
        weights = np.exp(-dists * rho)
    else:
        weights = np.exp(-dists / decay_m)
    weights[dists>max_dist_m] = 0
    np.fill_diagonal(weights, 0)

    demo_vals = grid[characteristic_var].fillna(0).values

    access = weights @ demo_vals

    grid = grid.copy()
    grid[access_name] = access
    grid['log_' + access_name] = np.log(grid[access_name] + 0.000000001)
    grid[access_name + '_sq'] = grid[access_name] ** 2

    return grid

def high_access_indicator(grid, access_var, threshold, city_specific = True):
    """Return a copy of grid with a new indicator column for having access_var above
    threshold."""
    grid = grid.copy()
    if city_specific:
        threshold = grid.groupby('city')[access_var].transform(lambda x: x.quantile(threshold))
    else:
        threshold = grid['access_var'].quantile(threshold)
    grid[f'high_{access_var}'] = (grid[access_var] >= threshold).astype(int)
    return grid

def assign_highway_exposure(df, high_exposure_threshold, low_exposure_threshold, hwy_col='hwy'):
    """Label each grid square by its distance to the nearest square where hwy_col == 1:
    within high_exposure_threshold is 'high' exposure, farther than that but within
    low_exposure_threshold is 'low' exposure, and farther than low_exposure_threshold is
    left unlabeled (None)."""
    hwy_centroids = df.loc[df[hwy_col] == 1].geometry.centroid
    if hwy_centroids.empty:
        raise ValueError(f"No squares found with {hwy_col} == 1")
    hwy_coords = np.column_stack([hwy_centroids.x.values, hwy_centroids.y.values])

    centroids = df.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    dist_to_hwy, _ = KDTree(hwy_coords).query(coords, k=1)

    df = df.copy()
    df['hwy_exposure'] = np.where(
        dist_to_hwy <= high_exposure_threshold, 'high',
        np.where(dist_to_hwy <= low_exposure_threshold, 'low', None)
    )
    df['high_exposure'] = (df['hwy_exposure'] == 'high').astype(int).fillna(0).astype(int)
    df['low_exposure'] = (df['hwy_exposure'] == 'low').astype(int).fillna(0).astype(int)
    return df

def widen_highways(grid, buffer_m):
    hwy_squares = grid.loc[grid['hwy'] == 1, ['grid_id', 'geometry']]
    corridor = gpd.GeoDataFrame(geometry=hwy_squares.buffer(buffer_m), crs=grid.crs)

    grid_geo = grid[['grid_id', 'geometry']]
    hwy_wide = gpd.sjoin(grid_geo, corridor, how='left', predicate='intersects')
    hwy_wide['hwy_wide'] = np.where(hwy_wide['index_right'].isna(), 0, 1)
    hwy_wide = hwy_wide.groupby('grid_id').agg({'hwy_wide': 'max'}).reset_index()
    grid = grid.merge(hwy_wide, on='grid_id', how='left')
    grid['hwy_wide'] = grid['hwy_wide'].fillna(0).astype(int)
    return grid


def impute_values(df, columns):
    imputed_mask = df['imputed'] == 1
    columns = [c for c in columns if df.loc[imputed_mask, c].isna().any()]
    touches = gpd.sjoin(df[['geometry']], df[['geometry']], how='inner', predicate='touches')

    def neighbor_median(col):
        neighbor_vals = df[col].to_numpy()[touches['index_right'].to_numpy()]
        medians = pd.Series(neighbor_vals, index=touches.index).groupby(level=0).median()
        result = pd.Series(np.nan, index=df.index)
        result.loc[medians.index] = medians.to_numpy()
        return result
    for col in columns:
        df.loc[imputed_mask, col] = df.loc[imputed_mask, col].fillna(neighbor_median(col))
    return df

def city_specific_shares(df, quantile = 0.25):
    """Compute an indicator for having a 'high share' of the Black population based on each city's own distribution, and return a copy of df with that column added."""
    df = df.copy()
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_df = df.loc[city_mask]
        threshold = city_df.loc[city_df['any_black'] == 1, 'share_black'].quantile(quantile)
        df.loc[city_mask, 'high_black_share'] = (city_df['share_black'] >= threshold).astype(int)
    return df

def compute_shares(df):
    df = df.copy()
    for city in df['city'].unique():
        city_mask = df['city'] == city
        city_df = df.loc[city_mask]
        total_homeowners = city_df['homeowners'].sum()
        total_black_pop = city_df['black_pop'].sum()
        share_black = np.where(total_black_pop > 0, city_df['black_pop'] / total_black_pop, 0)
        share_black_homeowners = np.where(city_df['black_homeowners'].sum() > 0, city_df['black_homeowners'] / city_df['black_homeowners'].sum(), 0)
        share_homeowners = np.where(city_df['homeowners'].sum() > 0, city_df['homeowners'] / total_homeowners, 0)
        df.loc[city_mask, 'share_black'] = share_black
        df.loc[city_mask, 'share_black_homeowners'] = share_black_homeowners
        df.loc[city_mask, 'share_homeowners'] = share_homeowners
    return df
