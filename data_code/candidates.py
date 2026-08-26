import pandas as pd
import numpy as np
import geopandas as gpd
import itertools
from shapely.geometry import LineString
import pickle
import scipy.stats as stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from pathlib import Path

# candidate list for CNN - only based on location
def create_mlcandidate_list(data, cbd):
    # get rays between each existing highway point
    pts = data.loc[data['hwy_40'] == 1].copy()
    if pts.empty:
        print('No exisiting highways found')
        return []

    # get lines between each existing highway ray
    centroids = pts.geometry.centroid.reset_index(drop = True)
    n = len(centroids)
    pairs = itertools.combinations(range(n), 2)

    lines = []
    for i, j in pairs:
        p1 = centroids.iloc[i]
        p2 = centroids.iloc[j]
        lines.append(LineString([(p1.x, p1.y), (p2.x, p2.y)]))

    # get rays between each existing highway ray and the CBD
    cbd_point = cbd.geometry.iloc[0]
    for p in centroids:
        lines.append(LineString([(p.x, p.y), (cbd_point.x, cbd_point.y)]))

    rays = gpd.GeoDataFrame(geometry = gpd.GeoSeries(lines, crs = data.crs))
    
    # get list of grid_ids that the rays intersect
    candidates = gpd.sjoin(data, rays, how = 'inner', predicate = 'intersects')
    
    # drop candidates that already have highways and those that will have highways
    candidates = candidates.loc[candidates['hwy_40'] == 0].copy()
    candidates = candidates.loc[candidates['hwy'] == 0].copy()
    return candidates['grid_id'].unique().tolist()

def get_mlcandidates(data, centroids, sample):
    candidate_list = {}
    for city in sample['city'].unique():
        city_data = data[data['city'] == city].copy()
        city_mask = centroids['place'].str.lower().str.replace(' ', '') == city.lower().replace(' ', '')
        if not city_mask.any():
            raise ValueError(f"No CBD centroid found for city '{city}' in centroids['place']")
        city_cbd = centroids[city_mask]
        candidate_list[city] = create_mlcandidate_list(city_data, city_cbd)
    out_path = Path('data/output/cnn_candidate_list.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(candidate_list, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return candidate_list

def compute_hwy_degree(df):
    """For each hwy_40 square, count how many other hwy_40 squares it touches
    (queen contiguity: shared edge or corner), and label which connected
    segment it belongs to. Degree 0/1 squares are segment endpoints (0 = an
    isolated single-square segment), degree 2 squares are segment interiors,
    and degree >= 3 squares are junctions where multiple segments meet.
    component_id groups squares that are touch-connected into the same
    segment (or junction cluster) -- e.g. to tell apart two endpoints that
    belong to the same segment from two that belong to different ones."""
    hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].reset_index(drop=True)
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

    n = len(hwy_40_squares)
    rows = matched.index.to_numpy()
    cols = matched['index_right'].to_numpy(dtype=int)
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(adjacency, directed=False)
    out['component_id'] = labels
    return out

# create a candidate list for non-ML based sample selection - include location and elevation
def create_candidate_list(data, cbd, buffer_width_m=0, k=2):
    """buffer_width_m widens each ray into a corridor of that width before intersecting
    the grid (buffer_width_m=0, the default, replicates plain line intersection) -- used
    by robustness/corridor_width.py to sweep corridor width while sharing this exact
    construction, so buffer_width_m=0 there is guaranteed identical to the candidate_dict
    built here rather than relying on a separately-maintained copy staying in sync.

    Rays are drawn from each highway segment endpoint to its k nearest endpoints
    belonging to a *different* segment (component_id) -- connecting every pair of
    endpoints regardless of distance produced candidates spanning implausibly large
    areas (e.g. opposite corners of a city)."""

    # identify highway endpoints using hwy degree < 2
    degree = compute_hwy_degree(data)
    data = data.merge(degree, on='grid_id', how='left')

    pts = data.loc[data['hwy_degree'] < 2].copy()
    if pts.empty:
        print('No exisiting highways found')
        return []

    # connect each endpoint to its k nearest endpoints on other segments
    centroids = pts.geometry.centroid.reset_index(drop=True)
    component_ids = pts['component_id'].reset_index(drop=True)
    coords = np.column_stack([centroids.x.values, centroids.y.values])
    n = len(coords)

    pairs = set()
    if n >= 2:
        tree = cKDTree(coords)
        _, idxs = tree.query(coords, k=n)  # full ranking of neighbors, nearest first
        for i, neighbor_idxs in enumerate(idxs):
            kept = 0
            for j in neighbor_idxs:
                if j == i or component_ids[j] == component_ids[i]:
                    continue
                pairs.add((min(i, j), max(i, j)))
                kept += 1
                if kept == k:
                    break

    lines = []
    for i, j in pairs:
        p1 = centroids.iloc[i]
        p2 = centroids.iloc[j]
        lines.append(LineString([(p1.x, p1.y), (p2.x, p2.y)]))

    if buffer_width_m > 0:
        lines = [line.buffer(buffer_width_m) for line in lines]

    rays = gpd.GeoDataFrame(geometry = gpd.GeoSeries(lines, crs = data.crs))

    # get list of grid_ids that the rays intersect
    candidates = gpd.sjoin(data, rays, how = 'inner', predicate = 'intersects')

    # drop candidates that already have highways
    candidates = candidates.loc[candidates['hwy_40'] == 0].copy()
    return candidates['grid_id'].unique().tolist()

def get_candidates(data, centroids, sample):
    candidate_list = {}
    for city in sample['city'].unique():
        city_data = data[data['city'] == city].copy()
        city_mask = centroids['place'].str.lower().str.replace(' ', '') == city.lower().replace(' ', '')
        if not city_mask.any():
            raise ValueError(f"No CBD centroid found for city '{city}' in centroids['place']")
        city_cbd = centroids[city_mask]
        candidate_list[city] = create_candidate_list(city_data, city_cbd)
    out_path = Path('data/output/candidate_list.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(candidate_list, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return candidate_list 


####################################################################################################
data = pd.read_pickle('data/output/sample.pkl')
sample = pd.read_pickle('data/input/samplelist.pkl')
centroids = pd.read_csv('data/input/msas_with_central_city_cbds.csv')
centroids = gpd.GeoDataFrame(centroids, geometry = gpd.points_from_xy(centroids.cbd_retail_long, centroids.cbd_retail_lat), 
                             crs = 'EPSG:4267') # best guess at CRS based off of projfinder.com
ml_candidate_dict = get_mlcandidates(data, centroids, sample)
candidate_dict = get_candidates(data, centroids, sample)

# print the stats for my sake
candidate_list = [item for sublist in candidate_dict.values() for item in sublist]
candidates = data.loc[data['grid_id'].isin(candidate_list)].copy()
hwys = data['hwy'].sum()
print(f'{candidates['hwy'].sum()} out of {hwys} highways are in candidate list ({100 * candidates['hwy'].sum() / hwys:.2f}%)')
print(f'{len(candidate_list)} candidate squares out of {len(data)} total squares ({100 * len(candidate_list) / len(data):.2f}%)')