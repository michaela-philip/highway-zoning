import pandas as pd
import geopandas as gpd
import itertools
from shapely.geometry import LineString
import pickle
import scipy.stats as stats
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
        city_cbd = centroids[centroids['place'].str.lower() == f'{city}'].to_crs(city_data.crs)
        candidate_list[city] = create_mlcandidate_list(city_data, city_cbd)
    out_path = Path('data/output/cnn_candidate_list.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        pickle.dump(candidate_list, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return candidate_list

# create a candidate list for non-ML based sample selection - include location and elevation
def create_candidate_list(data, cbd):
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

    # keep candidates within 1 z-score of demeaned elevation 
    elev_z = stats.zscore(candidates['dm_elevation'])
    candidates = candidates.loc[(elev_z > -1) & (elev_z < 1)].copy()
    
    # drop candidates that already have highways and those that will have highways
    candidates = candidates.loc[candidates['hwy_40'] == 0].copy()
    candidates = candidates.loc[candidates['hwy'] == 0].copy()
    return candidates['grid_id'].unique().tolist()

def get_candidates(data, centroids, sample):
    candidate_list = {}
    for city in sample['city'].unique():
        city_data = data[data['city'] == city].copy()
        city_cbd = centroids[centroids['place'].str.lower() == f'{city}'].to_crs(city_data.crs)
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