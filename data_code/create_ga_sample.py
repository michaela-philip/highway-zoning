import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry

### FUNCTION TO CLASSIFY GRID BASED ON ZONING ###
def classify_grid(df, grid):
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
    return output

### FUNCTION TO PLACE CENSUS INFO INTO GRID ###
def place_census(census, grid):
    census = gpd.GeoDataFrame(census, geometry = gpd.points_from_xy(census.longitude, census.latitude), 
                          crs = 'EPSG:4269') # census geocodes in NAD83 for some reason
    census = census.to_crs(grid.crs)
    census['black_pop'] = (census['black'] * census['numprec'])
    census_grid = grid.sjoin(census, how='left', predicate='contains')

    # calculate population and demographics in each grid square
    agg_funcs = {
        'numprec':'sum',
        'black_pop': 'sum',
        'rent' : 'median',
        'valueh': 'median'
    }
    census_grid = census_grid.dissolve(by='grid_id', aggfunc=agg_funcs)
    ('census data dissolved to grid')

    # calculate a few different definitions of 'majority black'
    census_grid['pct_black'] = census_grid['black_pop'] / census_grid['numprec']
    census_grid['share_black'] = census_grid['black_pop'] / (census_grid['black_pop'].sum())
    census_grid['mblack_mean_pct'] = np.where(census_grid['pct_black'] >= (census_grid['pct_black'].mean()), 1, 0)
    census_grid['mblack_median_pct'] = np.where(census_grid['pct_black'] >= (census_grid['pct_black'].median()), 1, 0)
    census_grid['mblack_mean_share'] = np.where(census_grid['share_black'] >= (census_grid['share_black'].mean()), 1, 0)
    census_grid['mblack_median_share'] = np.where(census_grid['share_black'] >= (census_grid['share_black'].median()), 1, 0)
    
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
    state = state59.overlay(state40, how = 'difference', keep_geom_type=False)
    state = state[~state.is_empty]
    us = us59.overlay(us40, how = 'difference', keep_geom_type = False)
    us = us[~us.is_empty]
    interstate = interstate[~interstate.is_empty]

    # combine all roads
    all_roads = pd.concat([interstate, state, us])
    print('roads combined')

    atl_grid_hwy = gpd.sjoin(grid, all_roads, how = 'left', predicate = 'intersects')

    # dummy variable for presence of highway
    atl_grid_hwy['hwy'] = np.where(atl_grid_hwy['type'].isna(), 0, 1)

    # aggregate by grid_id taking max value (if any highways exist, it is 1 no matter what)
    atl_grid_hwy = atl_grid_hwy.groupby('grid_id').agg({'hwy': 'max'})

    # merge in hwy indicator
    output = grid.merge(atl_grid_hwy['hwy'], left_on='grid_id', right_index=True)
    return output

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(zoning, census, state59, state40, us59, us40, interstate, gridsize):
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
    output = classify_grid(zoning, grid)
    print(output.columns,'zoning added to grid')

    # overlay census data on grid
    output = output.merge(place_census(census, output)[['grid_id', 'numprec', 'black_pop', 'rent', 'valueh', 'pct_black', 'share_black', 'mblack_mean_pct', 'mblack_median_pct', 'mblack_mean_share', 'mblack_median_share']],
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

interstate = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp')
state59 = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us59 = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')
state40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp')
us40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp')

# create sample with 200m x 200m grid squares
atl_sample = create_grid(zoning, census, state59, state40, us59, us40, interstate, 150)
atl_sample.to_pickle('data/output/atl_sample.pkl')