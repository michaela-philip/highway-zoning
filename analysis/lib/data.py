import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_sample(path='data/output/sample.pkl'):
    """Load the grid-square sample and construct the variables used across specifications."""
    df = pd.read_pickle(path)
    # df['rent'] = df['rent'].replace(0, 0.00001)
    # df['valueh'] = df['valueh'].replace(0, 0.00001)

    df['log_valueh'] = np.log(df['valueh'] + 0.00001) * df['valueh_avail']
    df['log_rent'] = np.log(df['rent'] + 0.00001) * df['rent_avail']
    df['city_louisville'] = (df['city'] == 'louisville').astype(int)
    df['city_littlerock'] = (df['city'] == 'littlerock').astype(int)
    df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
    df['log_dist_to_rr'] = np.log(df['dist_to_rr'])
    df['log_dist_to_rr_sq'] = df['log_dist_to_rr'] ** 2
    df['log_dist_to_hwy'] = np.log(df['dist_to_hwy'])

    df['ResidentialxBlack'] = df['Residential'] * df['mblack_1945def']
    df['ResidentialxBlack_pct'] = df['Residential'] * df['mblack_mean_pct']
    df['ResidentialxBlack_share'] = df['Residential'] * df['mblack_mean_share']

    df['log_black'] = np.log(df['pct_black'] + 0.000001)
    df['ResidentialxLogBlack'] = df['Residential'] * df['log_black']
    df['ResidentialxPctBlack'] = df['Residential'] * df['pct_black']

    df['any_black'] = np.where(df['black_pop'] != 0, 1, 0)
    df['ResidentialxAnyBlack'] = df['Residential'] * df['any_black']
    df['BlackxPctOwners'] = df['mblack_1945def'] * df['owner']
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


def compute_hwy_degree(df):
    """For each hwy_40 square, count how many other hwy_40 squares it touches
    (queen contiguity: shared edge or corner). Degree 0/1 squares are segment
    endpoints (0 = an isolated single-square segment), degree 2 squares are
    segment interiors, and degree >= 3 squares are junctions where multiple
    segments meet."""
    hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].copy()
    touches_result = gpd.sjoin(
        hwy_40_squares,
        hwy_40_squares[['geometry']],
        how='left',
        predicate='touches',
    )
    matched = touches_result[touches_result['index_right'].notna()]
    degree = matched.groupby('grid_id').size()

    out = hwy_40_squares[['grid_id']].copy()
    out['hwy_degree'] = out['grid_id'].map(degree).fillna(0).astype(int)
    return out


def diagonal_touch_share(df):
    """Among touching pairs of hwy_40 squares, report the share whose only
    shared geometry is a single point (a corner-only/diagonal touch) rather
    than a shared edge."""
    hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].copy()
    touches_result = gpd.sjoin(
        hwy_40_squares,
        hwy_40_squares[['geometry']],
        how='inner',
        predicate='touches',
    )
    # each unordered pair appears twice (A->B and B->A); keep one direction
    touches_result = touches_result[touches_result.index < touches_result['index_right']]

    left_geom = hwy_40_squares.loc[touches_result.index, 'geometry'].reset_index(drop=True)
    right_geom = hwy_40_squares.loc[touches_result['index_right'], 'geometry'].reset_index(drop=True)
    intersections = gpd.GeoSeries(left_geom).intersection(gpd.GeoSeries(right_geom))
    is_corner_only = intersections.geom_type.isin(['Point', 'MultiPoint'])

    n_pairs = len(is_corner_only)
    n_corner_only = int(is_corner_only.sum())
    share = n_corner_only / n_pairs if n_pairs else float('nan')
    return share, n_corner_only, n_pairs


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


def add_cnn_interactions(df):
    """Add interaction terms between the CNN-predicted highway probability and
    Residential/Black, used in specifications that condition on the CNN covariate."""
    df = df.copy()
    df['BlackxProbHwy'] = df['mblack_1945def'] * df['prob_hwy']
    df['ResidentialxProbHwy'] = df['Residential'] * df['prob_hwy']
    df['ResidentialxBlackxProbHwy'] = df['Residential'] * df['mblack_1945def'] * df['prob_hwy']
    df['BlackxLogHwy'] = df['mblack_1945def'] * df['logit_hwy']
    df['ResidentialxLogHwy'] = df['Residential'] * df['logit_hwy']
    df['ResidentialxBlackxLogHwy'] = df['Residential'] * df['mblack_1945def'] * df['logit_hwy']
    df['ResidentialxPredProb'] = df['Residential'] * df['logit_hwy']
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