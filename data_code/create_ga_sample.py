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

    # reclassify grids by zoning type (similar approach Noelkel et al. (2020) HOLC classification)
    df['area'] = df['geometry'].area
    df['area_res'] = np.where(df['zoning'] == 'residential', df['area'], 0)
    df['area_ind'] = np.where(df['zoning'] == 'industrial', df['area'], 0)
    df = df.groupby('grid_id').agg({'area': 'sum', 'area_res': 'sum', 'area_ind': 'sum'})
    df['pct_res'] = df['area_res'] / df['area']
    df['pct_ind'] = df['area_ind'] / df['area']

    # classify based on relative percentages (may not be ideal)
    df['maj_zoning'] = np.where(df['pct_res'] > df['pct_ind'], 'residential', 'industrial')
    return df

### FUNCTION TO PLACE CENSUS INFO INTO GRID ###
def place_census(census, grid):
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
    census_grid['pct_black'] = census_grid['black_pop'] / census_grid['numprec']
    census_grid['share_black'] = census_grid['black_pop'] / (census_grid['black_pop'].sum())
    return census_grid

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(zoning, census, gridsize):
    # grid is fit to size of zoning map
    a, b, c, d  = zoning.total_bounds
    step = gridsize # gridsize in meters

    grid = gpd.GeoDataFrame(geometry = [
    shapely.geometry.box(minx, miny, maxx, maxy)
    for minx, maxx in zip(np.arange(a, c, step), np.arange(a, c, step)[1:])
    for miny, maxy in zip(np.arange(b, d, step), np.arange(b, d, step)[1:])], crs = zoning.crs)

    # numeric id for each grid square to assist with aggregation
    grid['grid_id'] = range(1, len(grid) + 1)

    # overlay zoning map with grid squares and classify each square
    zoning = classify_grid(zoning, grid)

    # overlay census data on grid
    census = place_census(census, grid)

    # merge zoning and census data into grid
    output = grid.merge(zoning[['maj_zoning']], left_on = 'grid_id', right_index = True)
    output = output.merge(census, on='grid_id', how='left')
    return output

####################################################################################################

census = pd.read_csv('data/output/atl_geocoded.csv')
# census geocodes in NAD83 for some reason
census = gpd.GeoDataFrame(census, geometry = gpd.points_from_xy(census.longitude, census.latitude), 
                          crs = 'EPSG:4269')
zoning = gpd.read_file('data/input/zoning_shapefiles/atlanta/zoning.shp')

# create sample with 200m x 200m grid squares
atl_sample = create_grid(zoning, census, 200)