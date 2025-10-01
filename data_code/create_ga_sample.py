import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry

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
    city_cbd = centroids[centroids['place'].str.lower() == f'{city}'].to_crs(grid.crs).geometry.iloc[0]
    output['distance_to_cbd'] = output.geometry.centroid.distance(city_cbd)
    return output

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
    while True:
        census['prev_grid'] = census['grid_id'].shift(1)
        census['next_grid'] = census['grid_id'].shift(-1)
        candidates = (census['grid_id'].isna() & census['prev_grid'].notna() & census['next_grid'].notna() & (census['prev_grid'] == census['next_grid'])) #if i-1 and i+1 are in the same grid, assign i to that grid
        if candidates.any():
            candidate_serials.extend(census.loc[candidates, 'serial'].tolist())
            census.loc[candidates, 'grid_id'] = census.loc[candidates, 'prev_grid']
            print(candidates.sum(), 'candidates')
        else:
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
        'serial': 'count'
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
    hwy59 = gpd.sjoin(grid, built_1959, how = 'left', predicate = 'intersects')
    hwy_40 = gpd.sjoin(grid, built_1940, how = 'left', predicate = 'intersects')

    # dummy variable for presence of highway
    hwy59['hwy_59'] = np.where(hwy59['speed1'].isna(), 0, 1)
    hwy_40['hwy_40'] = np.where(hwy_40['speed1'].isna(), 0, 1)

    # aggregate by grid_id taking max value (if any highways exist, it is 1 no matter what)
    hwy59 = hwy59.groupby('grid_id').agg({'hwy_59':'max'})
    hwy_40 = hwy_40.groupby('grid_id').agg({'hwy_40':'max'})

    hwys = pd.concat([hwy59, hwy_40], axis = 1)

    # merge in hwy indicator
    output = grid.merge(hwys, left_on='grid_id', right_index=True)
    return output

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(zoning, centroids, census, state59, state40, us59, us40, interstate, gridsize, city_sample, zoning2 = None):
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
    output = classify_grid(zoning, grid, centroids, city_sample, zoning2)
    print('zoning added to grid')

    # overlay census data on grid
    output = output.merge(place_census(census, output)[['grid_id', 'numprec', 'black_pop', 'rent', 'valueh', 
                                                                  'pct_black', 'share_black', 'mblack_mean_pct', 
                                                                  'mblack_mean_share', 'mblack_1945def', 'serial']],
                           on='grid_id', how='left')
    print('census added to grid')

    # place highways into grid
    output = output.merge(place_highways(grid, state59, state40, us59, us40, interstate)[['grid_id', 'hwy_59', 'hwy_40']],
                            on='grid_id',how='left')
    print('highways added to grid')

    # difference hwy indicator at the grid level
    output['hwy'] = output['hwy_59'] - output['hwy_40']
    output['hwy'] = np.where(output['hwy'] < 0, 0, output['hwy'])
    output = output[output['numprec'] > 0]
    return output

def create_sample(df, sample):
    output = pd.DataFrame()
    for city in sample['city'].unique():
        city_sample = sample[sample['city'] == city].iloc[0]
        city_df = df[df['city'] == city].copy()
        if city == 'louisville':
            city_zoning1 = zoning['louisville_1947']
            city_zoning2 = zoning['louisville_1931']
            city_grid = create_grid(city_zoning1, centroids, city_df, state59, state40, us59, us40, interstate, gridsize = 150, city_sample = city_sample, zoning2 = city_zoning2)
        else:
            city_zoning = zoning[city]
            city_grid = create_grid(city_zoning, centroids, city_df, state59, state40, us59, us40, interstate, city_sample = city_sample, gridsize = 150)
        city_grid['city'] = city
        output = pd.concat([output, city_grid], ignore_index=True)
    return output

####################################################################################################
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

####################################################################################################
### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
atlanta_zoning = gpd.read_file('data/input/zoning_shapefiles/atlanta/zoning.shp')
louisville_zoning1 = gpd.read_file('data/input/zoning_shapefiles/louisville/zoning-1947.shp')
louisville_zoning2 = gpd.read_file('data/input/zoning_shapefiles/louisville/zoning-1931.shp')
zoning = {
    'atlanta': atlanta_zoning,
    'louisville_1947': louisville_zoning1,
    'louisville_1931': louisville_zoning2
}
values = [
    ('atlanta', 'AT', 'georgia', 'GA', 44, [1210, 890], 350),
    ('louisville', 'LO', 'kentucky', 'KY', 51, [1110], 3750)]
keys=['city', 'cityabbr', 'state', 'stateabbr', 'stateicp', 'countyicp', 'cityicp']
rows = [dict(zip(keys, v)) for v in values]
sample = pd.DataFrame(rows)
####################################################################################################

# create sample with 150 x 150 grid squares
output = create_sample(census, sample)
output.to_pickle('data/output/sample.pkl')