import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import shapely.geometry

def load_sample(size, impute = False):
    """Load the grid-square sample and construct the variables used across specifications."""
    path=f'data/output/sample_{size}.pkl'
    df = pd.read_pickle(path)
    if impute:
        df = impute_values(df, columns=df.columns)
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

    df['mblack_1945def'] = np.where(df['pct_black'] > 0.6, 1, 0)
    df['mblack_50_pct'] = np.where(df['pct_black'] > 0.5, 1, 0)
    df['mblack_40_pct'] = np.where(df['pct_black'] > 0.4, 1, 0)
    df['mblack_pct_mean'] = np.where(df['pct_black'] > df['pct_black'].mean(), 1, 0)
    df['mblack_share_mean'] = np.where(df['share_black'] > df['share_black'].mean(), 1, 0)
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

def compute_demographic_access(grid, demographic_var, decay_m, rho = None, max_dist_m = 5000):
    centroids = grid.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    dists = cdist(coords, coords, metric = 'euclidean')

    # compute distance decay weights
    weights = np.exp(-dists / decay_m)
    if rho is not None:
        weights = np.exp(-dists * rho)
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
    grid['log_dem_access'] = np.log(grid['dem_access_raw'])
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