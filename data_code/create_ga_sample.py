import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry
from intervaltree import IntervalTree

### FUNCTION TO CLASSIFY GRID BASED ON ZONING ###
def classify_grid(df, grid, centroids):
    # condense zoning to residential and industrial
    zoning_map = {'dwelling': 'residential', 'apartment': 'residential', 
                  'industrial': 'industrial', 'business': 'industrial'}
    df['zoning'] = df['Zonetype'].map(zoning_map)
    df = df[df['zoning'].notna()]
    df = df.overlay(grid, how = 'identity')
    print('zoning overlaid')

    # reclassify grids by zoning type (similar approach Noelkel et al. (2020) HOLC classification)
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

    # calculate distance between grid centroid and CBD 
    atl_cbd = centroids[centroids['place'] == 'Atlanta'].to_crs(grid.crs).geometry.iloc[0]
    output['distance_to_cbd'] = output.geometry.centroid.distance(atl_cbd)
    return output

### FUNCTION TO PLACE CENSUS INFO INTO GRID ###
def place_census(census, grid, geocoded):
    census = gpd.GeoDataFrame(census, geometry = gpd.points_from_xy(census.longitude, census.latitude), 
                            crs = 'EPSG:4269') # census geocodes in NAD83 for some reason
    census = census.to_crs(grid.crs)
    census['black_pop'] = (census['black'] * census['numprec'])
    census_grid = grid.sjoin(census, how='left', predicate='contains')

    # pull in households that weren't geocoded but are likely to exist in each grid
    grid = census_grid.groupby('grid_id').agg({'serial': ['min', 'max']}).reset_index().dropna()
    grid.columns = ['_'.join([str(i) for i in col if i]) for col in grid.columns.values]

    # create interval tree to find serials that fall within each grid's serial range
    tree = IntervalTree()
    for _, row in grid.iterrows():
        tree[int(row['serial_min']): int(row['serial_max']) + 1] = int(row['grid_id'])

    def assign_grid_id(serial):
        matches = tree[serial]
        if len(matches) == 1:
            return list(matches)[0].data
        else:
            return np.nan
    
    geocoded['serial'] = geocoded['serial'].astype(int)
    geocoded['grid_id'] = geocoded['serial'].apply(assign_grid_id)
    census_grid = pd.concat([census_grid, geocoded[geocoded['grid_id'].notna()]], ignore_index=True)
    
    # calculate population and demographics in each grid square
    agg_funcs = {
        'numprec':'sum',
        'black_pop': 'sum',
        'rent' : 'median',
        'valueh': 'median',
        'serial': 'count'
    }
    census_grid = census_grid.dissolve(by='grid_id', aggfunc=agg_funcs)
    print('census data dissolved to grid', census_grid.columns)
    #census_grid = census_grid.reset_index()

    # calculate a few different definitions of 'majority black'
    census_grid['pct_black'] = census_grid['black_pop'] / census_grid['numprec']
    census_grid['share_black'] = census_grid['black_pop'] / (census_grid['black_pop'].sum())
    census_grid['mblack_mean_pct'] = np.where(census_grid['pct_black'] >= (census_grid['pct_black'].mean()), 1, 0)
    census_grid['mblack_mean_share'] = np.where(census_grid['share_black'] >= (census_grid['share_black'].mean()), 1, 0)
    census_grid['mblack_1945def'] = np.where(census_grid['pct_black'] >= 0.60, 1, 0)
    
    output = grid.merge(census_grid, left_on='grid_id', right_index=True)
    return output

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

    # include only roads that exist in 1959 and did not exist in 1940
    state_overlap = gpd.sjoin(state59, state40, how = 'inner', predicate = 'intersects')
    us_overlap = gpd.sjoin(us59, us40, how = 'inner', predicate = 'intersects')
    state = state59.overlay(state_overlap, how = 'difference', keep_geom_type=False)
    us = us59.overlay(us_overlap, how = 'difference', keep_geom_type = False)
    
    state = state[~state.is_empty]
    us = us[~us.is_empty]
    interstate = interstate[~interstate.is_empty]

    # combine all roads
    all_roads = pd.concat([interstate, state, us])
    print('roads combined')

    # merge into grid
    atl_grid_hwy = gpd.sjoin(grid, all_roads, how = 'left', predicate = 'intersects')

    # dummy variable for presence of highway
    atl_grid_hwy['hwy'] = np.where(atl_grid_hwy['speed1'].isna(), 0, 1)

    # aggregate by grid_id taking max value (if any highways exist, it is 1 no matter what)
    atl_grid_hwy = atl_grid_hwy.groupby('grid_id').agg({'hwy': 'max'})

    # merge in hwy indicator
    output = grid.merge(atl_grid_hwy['hwy'], left_on='grid_id', right_index=True)
    return output

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(zoning, centroids, census, geocoded, state59, state40, us59, us40, interstate, gridsize):
    # grid is fit to size of zoning map
    a, b, c, d  = zoning.total_bounds
    step = gridsize # gridsize in meters

    grid = gpd.GeoDataFrame(geometry = [
    shapely.geometry.box(minx, miny, maxx, maxy)
    for minx, maxx in zip(np.arange(a, c, step), np.arange(a, c, step)[1:])
    for miny, maxy in zip(np.arange(b, d, step), np.arange(b, d, step)[1:])], crs = zoning.crs)
    print('grid created')

    # numeric id for each grid square to assist with aggregation
    grid['grid_id'] = range(1, len(grid) + 1)

    # overlay zoning map with grid squares and classify each square
    output = classify_grid(zoning, grid, centroids)
    print(output.columns,'zoning added to grid')

    # overlay census data on grid
    output = output.merge(place_census(census, output, geocoded)[['grid_id', 'numprec', 'black_pop', 'rent', 'valueh', 
                                                                  'pct_black', 'share_black', 'mblack_mean_pct', 
                                                                  'mblack_mean_share', 'mblack_1945def', 'serial']],
                           on='grid_id', how='left')
    print(output.columns, 'census added to grid')

    # place highways into grid
    output = output.merge(place_highways(grid, state59, state40, us59, us40, interstate)[['grid_id', 'hwy']],
                           on='grid_id',how='left')
    print(output.columns,'highways added to grid')
    return output

####################################################################################################

census = pd.read_pickle('data/input/atl_geocoded.pkl')
zoning = gpd.read_file('data/input/zoning_shapefiles/atlanta/zoning.shp')

geocoded = pd.read_pickle('data/input/atl_geocoded.pkl')
geocoded = geocoded[geocoded['is_exact'].isna()].copy()

centroids = pd.read_csv('data/input/msas_with_central_city_cbds.csv') # from Dan Aaron Hartley
centroids = gpd.GeoDataFrame(centroids, geometry = gpd.points_from_xy(centroids.cbd_retail_long, centroids.cbd_retail_lat), 
                             crs = 'EPSG:4267') # best guess at CRS based off of projfinder.com

interstate = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp') # from Jaworski and Kitchens
state59 = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us59 = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')
state40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp')
us40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp')

# create sample with 150 x 150 grid squares
atl_sample = create_grid(zoning, centroids, census, geocoded, state59, state40, us59, us40, interstate, 150)
atl_sample.to_pickle('data/output/atl_sample.pkl')