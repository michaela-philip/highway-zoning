import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry
import requests
import time

####################################################################################################
### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
atlanta_zoning = gpd.read_file('data/input/zoning_shapefiles/atlanta/zoning.shp')
louisville_zoning1 = gpd.read_file('data/input/zoning_shapefiles/louisville/zoning-1947.shp')
louisville_zoning2 = gpd.read_file('data/input/zoning_shapefiles/louisville/zoning-1931.shp')
littlerock = gpd.read_file('data/input/zoning_shapefiles/littlerock/zoning-1937.shp')
zoning = {
    'atlanta': atlanta_zoning,
    'louisville_1947': louisville_zoning1,
    'louisville_1931': louisville_zoning2,
    'littlerock': littlerock
}
atlanta_geology = gpd.read_file('data/input/atlanta/water_topog.shp')
louisville_geology = gpd.read_file('data/input/louisville/water_topog.shp')
littlerock_geology = gpd.read_file('data/input/littlerock/water_topog/water_topog.shp')
geology = {
    'atlanta': atlanta_geology,
    'louisville': louisville_geology,
    'littlerock': littlerock_geology
}

####################################################################################################
### FUNCTION TO CLASSIFY GRID BASED ON ZONING ###
def classify_grid(zoning1, grid, centroids, city_sample, zoning2 = None):
    # condense zoning to residential and industrial
    zoning_map = {'dwelling': 'residential', 'apartment': 'residential', 
                    'single family': 'residential', '2 family': 'residential', 
                    '2-4 family': 'residential', 'commercial': 'industrial',
                    'industrial': 'industrial', 'business': 'industrial',
                    'light industry': 'industrial', 'heavy industry': 'industrial'
                    }
    zoning1['zoning'] = zoning1['Zonetype'].map(zoning_map)
    zoning1 = zoning1[zoning1['zoning'].notna()]
    zoning1 = zoning1.overlay(grid, how = 'identity')
    print('zoning 1 overlaid')

    # reclassify grids by zoning type (similar approach Noelkel et al. (2020) HOLC classification)
    def classify_grids(df):
        df['area'] = df['geometry'].area
        df['area_res'] = np.where(df['zoning'] == 'residential', df['area'], 0)
        df['area_ind'] = np.where(df['zoning'] == 'industrial', df['area'], 0)
        df = df.groupby('grid_id').agg({'area': 'sum', 'area_res': 'sum', 'area_ind': 'sum'})
        df['pct_res'] = df['area_res'] / df['area']
        df['pct_ind'] = df['area_ind'] / df['area']

        # classify based on relative percentages (may not be ideal)
        df['maj_zoning'] = np.where(df['pct_res'] > df['pct_ind'], 'residential', 'industrial')
        output = grid.merge(df['maj_zoning'], left_on='grid_id', right_index=True)
        output['Residential'] = np.where(output['maj_zoning'] == 'residential', 1, 0)
        return output

    output = classify_grids(zoning1)

    if zoning2 is not None:
        print('using two zoning maps')
        zoning2['zoning'] = zoning2['Zonetype'].map(zoning_map)
        zoning2 = zoning2[zoning2['zoning'].notna()]
        zoning2 = zoning2.overlay(grid, how = 'identity')
        output2 = classify_grids(zoning2)

        # drop all squares whose zoning classification changes
        # compare only squares that exist in both maps - squares that only come into existence later are left as is
        overlap = output[['grid_id', 'maj_zoning']].merge(output2[['grid_id', 'maj_zoning']], 
                                                            on='grid_id', suffixes=('_1', '_2'), how = 'inner')
        drop_ids = overlap.loc[overlap['maj_zoning_1'] != overlap['maj_zoning_2'], 'grid_id']
        output = output[~output['grid_id'].isin(drop_ids)]
        print(len(drop_ids), 'grid squares dropped due to zoning change')

    city = city_sample['city']
    # calculate distance between grid centroid and CBD 
    city_mask = centroids['place'].str.lower().str.replace(' ', '') == city.lower().replace(' ', '')
    if not city_mask.any():
        raise ValueError(f"No CBD centroid found for city '{city}' in centroids['place']")
    city_cbd = centroids[city_mask].to_crs(grid.crs).geometry.iloc[0]
    output['distance_to_cbd'] = output.geometry.centroid.distance(city_cbd)
    return output

### FUNCTION TO OVERLAY GEOLOGICAL INFO ON GRID ###
def place_geology(geology, grid):
    geology = geology.to_crs(grid.crs)
    geology_grid = gpd.sjoin(grid, geology, how = 'left', predicate = 'contains')
    geology_grid = geology_grid.rename(columns = {'RASTERVALU':'elevation'})
    geology_grid = geology_grid.dissolve(by='grid_id', aggfunc={'elevation':'mean', 'dist_water':'mean'})
    return geology_grid

def _arcgis_query(url, params, retries=5, backoff=2):
    for attempt in range(retries):
        response = requests.get(url, params=params)
        if response.status_code in (502, 503, 504):
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
                continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()

### FUNCTION TO ADD FLOOD PLAIN DATA TO GRID ###
FLOOD_HAZARD_URL = "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query"
RAILROAD_URL = "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/Transportation_v1/FeatureServer/9/query"

def place_floodplains(grid):
    # get bounding box in WGS84 for the API spatial filter
    bounds = grid.to_crs('EPSG:4326').total_bounds  # (minx, miny, maxx, maxy)
    envelope = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"

    # paginate through results (server max is 750 per request)
    all_features = []
    offset = 0
    while True:
        params = {
            'where': '1=1',
            'geometry': envelope,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'SFHA_TF',
            'returnGeometry': 'true',
            'outSR': '3857',
            'f': 'geojson',
            'resultOffset': offset,
            'resultRecordCount': 750,
        }
        data = _arcgis_query(FLOOD_HAZARD_URL, params)
        features = data.get('features', [])
        all_features.extend(features)
        if not data.get('properties', {}).get('exceededTransferLimit', False):
            break
        offset += 750
    print(f'flood plains: fetched {len(all_features)} features')

    if not all_features:
        grid['flood_risk'] = 0
        return grid[['grid_id', 'flood_risk']]

    flood_gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:3857')
    flood_gdf = flood_gdf.to_crs(grid.crs)

    # keep only SFHA == True polygons
    sfha = flood_gdf[flood_gdf['SFHA_TF'] == 'T']

    # any grid cell intersecting an SFHA polygon gets flood_risk = 1
    joined = gpd.sjoin(grid[['grid_id', 'geometry']], sfha[['geometry']], how='left', predicate='intersects')
    flood_risk = joined.groupby('grid_id')['index_right'].any().astype(int).rename('flood_risk')
    output = grid[['grid_id']].merge(flood_risk, on='grid_id', how='left')
    output['flood_risk'] = output['flood_risk'].fillna(0).astype(int)
    return output

### FUNCTION TO ADD RAILROAD DISTANCE TO GRID ###
def place_railroads(grid):
    # buffer bounding box by ~1km (in degrees) to catch railroads just outside grid edge
    bounds = grid.to_crs('EPSG:4326').total_bounds  # (minx, miny, maxx, maxy)
    buf = 0.01
    envelope = f"{bounds[0]-buf},{bounds[1]-buf},{bounds[2]+buf},{bounds[3]+buf}"

    all_features = []
    offset = 0
    while True:
        params = {
            'where': '1=1',
            'geometry': envelope,
            'geometryType': 'esriGeometryEnvelope',
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelIntersects',
            'outFields': 'OBJECTID',
            'returnGeometry': 'true',
            'outSR': '3857',
            'f': 'geojson',
            'resultOffset': offset,
            'resultRecordCount': 2000,
        }
        data = _arcgis_query(RAILROAD_URL, params)
        features = data.get('features', [])
        all_features.extend(features)
        if not data.get('properties', {}).get('exceededTransferLimit', False):
            break
        offset += 2000
    print(f'railroads: fetched {len(all_features)} features')

    if not all_features:
        grid['dist_to_rr'] = np.nan
        return grid[['grid_id', 'dist_to_rr']]

    rr_gdf = gpd.GeoDataFrame.from_features(all_features, crs='EPSG:3857')
    rr_gdf = rr_gdf.to_crs(grid.crs)
    rr_union = rr_gdf.union_all()

    output = grid[['grid_id', 'geometry']].copy()
    output['dist_to_rr'] = output.geometry.centroid.apply(lambda x: x.distance(rr_union))
    return output[['grid_id', 'dist_to_rr']]

### FUNCTION TO PLACE CENSUS INFO INTO GRID ###
def place_census(census, grid):
    mask = (census['coordinates'].notna() & census['longitude'].isna())
    census.loc[mask, 'longitude'] = census.loc[mask, 'coordinates'].apply(lambda x: x[0])
    census.loc[mask, 'latitude'] = census.loc[mask, 'coordinates'].apply(lambda x: x[1])
    census = gpd.GeoDataFrame(census, geometry = gpd.points_from_xy(census.longitude, census.latitude), 
                            crs = 'EPSG:4269') # census geocodes in NAD83 for some reason
    census = census.to_crs(grid.crs)
    census['black_pop'] = (census['black'] * census['numprec'])
    census_grid = grid.sjoin(census, how='left', predicate='contains')
    print(census_grid.describe())

    # similar to while geocoding, I will interpolate grid_id by comparing neighbors
    census = census.merge(census_grid[['serial', 'grid_id']], on = 'serial', how = 'left')
    candidate_serials = []
    iter_count = 0
    max_iters = 5
    while True:
        census['prev_grid'] = census['grid_id'].shift(1)
        census['next_grid'] = census['grid_id'].shift(-1)
        candidates = (census['grid_id'].isna() & census['prev_grid'].notna() & census['next_grid'].notna() & (census['prev_grid'] == census['next_grid'])) #if i-1 and i+1 are in the same grid, assign i to that grid
        if candidates.any():
            candidate_serials.extend(census.loc[candidates, 'serial'].tolist())
            census.loc[candidates, 'grid_id'] = census.loc[candidates, 'prev_grid'].values
            print(candidates.sum(), 'candidates')
            iter_count += 1
        else:
            break
        if iter_count >= max_iters:
            print('max iters reached')
            break
    census = census.drop(columns = ['prev_grid', 'next_grid'])
    census = census[census['serial'].isin(candidate_serials)]
    
    # add in additional households to the grid
    census_grid = pd.concat([census_grid, census], ignore_index=True)
    
    # calculate population and demographics in each grid square
    agg_funcs = {
        'numprec':'sum',
        'black_pop': 'sum',
        'rent' : 'median',
        'valueh': 'median',
        'serial': 'count',
        'owner':'mean'
    }
    census_grid = census_grid.dissolve(by='grid_id', aggfunc=agg_funcs)
    print('census data dissolved to grid')

    # calculate a few different definitions of 'majority black'
    census_grid['pct_black'] = census_grid['black_pop'] / census_grid['numprec']
    census_grid['share_black'] = census_grid['black_pop'] / (census_grid['black_pop'].sum())
    census_grid['mblack_mean_pct'] = np.where(census_grid['pct_black'] >= (census_grid['pct_black'].mean()), 1, 0)
    census_grid['mblack_mean_share'] = np.where(census_grid['share_black'] >= (census_grid['share_black'].mean()), 1, 0)
    census_grid['mblack_1945def'] = np.where(census_grid['pct_black'] >= 0.60, 1, 0)
    
    output = grid.merge(census_grid, left_on='grid_id', right_index=True)
    return output

### FUNCTION TO INTERPOLATE MISSING RENT AND HOME VALUES ###
def impute_values(df):
    df = df.copy()
    sindex = df.sindex

    # imputed variable
    df['imputed_rent'] = np.nan
    df['imputed_valueh'] = np.nan

    neighbors_dict = {}
    for idx, geom in df['geometry'].items():
        possible_matches_index = list(sindex.intersection(geom.bounds))
        possible_matches = df.iloc[possible_matches_index]
        neighbors = possible_matches[possible_matches['geometry'].touches(geom)]
        neighbors_dict[idx] = neighbors.index.tolist()

    for idx in df.index:
        neighbor_idxs = neighbors_dict[idx]
        neighbor_rents = df.loc[neighbor_idxs, 'rent'].dropna()
        if not neighbor_rents.empty:
            df.at[idx, 'imputed_rent'] = neighbor_rents.mean()
    
    for idx in df.index:
        neighbor_idxs = neighbors_dict[idx]
        neighbor_values = df.loc[neighbor_idxs, 'valueh'].dropna()
        if not neighbor_values.empty:
            df.at[idx, 'imputed_valueh'] = neighbor_values.mean()

    df['rent_avail'] = np.where(df['rent'].notna(), 1, 0)
    df['valueh_avail'] = np.where(df['valueh'].notna(), 1, 0)
    df['rent'] = df['rent'].fillna(0)
    df['valueh'] = df['valueh'].fillna(0)

    return df

### FUNCTION TO CLEAN HWY DATA AND ADD INTO GRID ###
def place_highways(grid, state59, state40, us59, us40, interstate):
    state59 = state59.to_crs(grid.crs)
    state40 = state40.to_crs(grid.crs)
    us59 = us59.to_crs(grid.crs)
    us40 = us40.to_crs(grid.crs)
    interstate = interstate.to_crs(grid.crs)

    # fclass information from Baik et al. (2010)
    # eliminate all local roads
    state59 = state59[~state59['FCLASS'].isin([9,19])]
    state40 = state40[~state40['FCLASS'].isin([9,19])]
    us59 = us59[~us59['FCLASS'].isin([9,19])]
    us40 = us40[~us40['FCLASS'].isin([9,19])]
    interstate = interstate[~interstate['FCLASS'].isin([9,19])]

    # combine all types of roads for the bookends of our sample period
    built_1959  = pd.concat([state59, us59, interstate])
    built_1940 = pd.concat([state40, us40])

    # overlay highways onto grid
    hwy_59 = gpd.sjoin(grid, built_1959, how = 'left', predicate = 'intersects')
    hwy_40 = gpd.sjoin(grid, built_1940, how = 'left', predicate = 'intersects')

    # dummy variable for presence of highway
    hwy_59['hwy_59'] = np.where(hwy_59['speed1'].isna(), 0, 1)
    hwy_40['hwy_40'] = np.where(hwy_40['speed1'].isna(), 0, 1)

    # aggregate by grid_id taking max value (if any highways exist, it is 1 no matter what)
    hwy_59 = hwy_59.groupby('grid_id').agg({'hwy_59':'max'})
    hwy_40 = hwy_40.groupby('grid_id').agg({'hwy_40':'max'})

    hwys = pd.concat([hwy_59, hwy_40], axis = 1)

    # merge in hwy indicator
    output = grid.merge(hwys, left_on='grid_id', right_index=True)
    built_1940_union = built_1940.union_all()
    output['dist_to_hwy'] = output.geometry.centroid.apply(lambda x: x.distance(built_1940_union))
    return output

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(zoning, centroids, geology, census, state59, state40, us59, us40, interstate, gridsize, city_sample, zoning2 = None, grid_0 = 1):
    # grid is fit to size of zoning map
    a, b, c, d  = zoning.total_bounds
    step = gridsize # gridsize in meters

    grid = gpd.GeoDataFrame(geometry = [
    shapely.geometry.box(minx, miny, maxx, maxy)
    for minx, maxx in zip(np.arange(a, c, step), np.arange(a, c, step)[1:])
    for miny, maxy in zip(np.arange(b, d, step), np.arange(b, d, step)[1:])], crs = zoning.crs)
    print('grid created')

    # numeric id for each grid square to assist with aggregation
    grid['grid_id'] = range(grid_0, grid_0 + len(grid))

    # overlay zoning map with grid squares and classify each square
    output = classify_grid(zoning, grid, centroids, city_sample, zoning2)
    print('zoning added to grid')
    

    # overlay geological info 
    output = output.merge(place_geology(geology, output)[['dist_water', 'elevation']], on = 'grid_id', how = 'left')
    print('geology added to grid')

    # overlay census data on grid
    output = output.merge(place_census(census, output)[['grid_id', 'numprec', 'black_pop', 'rent', 'valueh', 
                                                                  'pct_black', 'share_black', 'mblack_mean_pct', 
                                                                  'mblack_mean_share', 'mblack_1945def', 'serial', 'owner']],
                           on='grid_id', how='left')
    print('census added to grid')

    # interpolate missing rent and home values
    output = impute_values(output)
    print('missing rent and home values imputed')

    # overlay flood plain data
    output = output.merge(place_floodplains(grid), on='grid_id', how='left')
    output['flood_risk'] = output['flood_risk'].fillna(0).astype(int)
    print('flood plains added to grid')

    # calculate distance to nearest railroad
    output = output.merge(place_railroads(grid), on='grid_id', how='left')
    print('railroad distances added to grid')

    # place highways into grid
    output = output.merge(place_highways(grid, state59, state40, us59, us40, interstate)[['grid_id', 'hwy_59', 'hwy_40', 'dist_to_hwy']],
                            on='grid_id',how='left')
    print('highways added to grid')

    # difference hwy indicator at the grid level
    output['hwy'] = output['hwy_59'] - output['hwy_40']
    output['hwy'] = np.where(output['hwy'] < 0, 0, output['hwy'])

    # bits and pieces
    output = output[output['numprec'] > 0]
    avg_elev = output.loc[output['hwy'] == 1, 'elevation'].mean()
    print('average elevation in hwy grids:', avg_elev)
    output['dm_elevation'] = output['elevation'] - avg_elev
    return output

def create_sample(df, sample):
    output = pd.DataFrame()
    grid_0 = 1
    for city in sample['city'].unique():
        city_sample = sample[sample['city'] == city].iloc[0]
        city_df = df[df['city'] == city].copy()
        city_geology = geology[city]
        if city == 'louisville':
            city_zoning1 = zoning['louisville_1947']
            city_zoning2 = zoning['louisville_1931']
            city_grid = create_grid(city_zoning1, centroids, city_geology, city_df, state59, state40, us59, us40, interstate, gridsize = 150, city_sample = city_sample, zoning2 = city_zoning2, grid_0 = grid_0)
        else:
            city_zoning = zoning[city]
            city_grid = create_grid(city_zoning, centroids, city_geology, city_df, state59, state40, us59, us40, interstate, city_sample = city_sample, gridsize = 150, grid_0 = grid_0)
        city_grid['city'] = city
        output = pd.concat([output, city_grid], ignore_index=True)
        grid_0 = output['grid_id'].max() + 1
    return output

####################################################################################################
### LOAD DATA ###
census = pd.read_pickle('data/intermed/geocoded_data.pkl')
centroids = pd.read_csv('data/input/msas_with_central_city_cbds.csv') # from Dan Aaron Hartley
centroids = gpd.GeoDataFrame(centroids, geometry = gpd.points_from_xy(centroids.cbd_retail_long, centroids.cbd_retail_lat), 
                             crs = 'EPSG:4267') # best guess at CRS based off of projfinder.com

interstate = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp') # from Jaworski and Kitchens
state59 = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us59 = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')
state40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp')
us40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp')

####################################################################################################
# create sample with 150 x 150 grid squares
sample = pd.read_pickle('data/input/samplelist.pkl')
output = create_sample(census, sample)
output.to_pickle('data/output/sample.pkl')