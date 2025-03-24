import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry

# all shapefiles courtesy of Professors Taylor Jaworski and Carl Kitchens
interstates = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp')
state_paved = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us_paved = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')

state_paved_40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp')
us_paved_40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp')

# change crs to ESPG 3857
interstates = interstates.to_crs('EPSG:3857')
state_paved = state_paved.to_crs('EPSG:3857')
us_paved = us_paved.to_crs('EPSG:3857')
state_paved_40 = state_paved_40.to_crs('EPSG:3857')
us_paved_40 = us_paved_40.to_crs('EPSG:3857')

# fclass information from Baik et al. (2010)
# eliminating all local roads
interstates = interstates[~interstates['FCLASS'].isin([9, 19])]
state_paved = state_paved[~state_paved['FCLASS'].isin([9,19])]
us_paved = us_paved[~us_paved['FCLASS'].isin([9,19])]
state_paved_40 = state_paved_40[~state_paved_40['FCLASS'].isin([9,19])]
us_paved_40 = us_paved_40[~us_paved_40['FCLASS'].isin([9,19])]

# include only roads that exist in 1959 and did not exist in 1940
# need to make sure that the order is correct to subtract _40 from _59
state_paved_new = gpd.overlay(state_paved_40, state_paved, how = 'difference')
state_paved_new = state_paved_new[~state_paved_new.is_empty]
us_paved_new = gpd.overlay(us_paved_40, us_paved, how = 'difference')
us_paved_new = us_paved_new[~us_paved_new.is_empty]

# add designation column in case we want to know which original csv it came from
interstates['designation'] = 'interstate'
state_paved['designation'] = 'state_paved'
us_paved['designation'] = 'us_paved'

all_roads = pd.concat([interstates, state_paved, us_paved])
print('concat done', all_roads.shape, all_roads.columns)

# read in atlanta zoning map
atl_zone = gpd.read_file('data/input/zoning_shapefiles/atlanta/zoning.shp')

# make a new zoning variable condensing only to residential and industrial
atl_zone['zoning'] = np.where(atl_zone['Zonetype'] == 'dwelling', 'residential', 
                              np.where(atl_zone['Zonetype'] == 'apartment', 'residential', 
                                       np.where(atl_zone['Zonetype'] == 'industrial', 'industrial', 
                                                np.where(atl_zone['Zonetype'] == 'business', 'industrial', 'nan'))))
atl_zone = atl_zone[~atl_zone['zoning'].isin(['nan'])]

# create grid of 0.5km x 0.5km squares overlaying the zoning map
a, b, c, d  = atl_zone.total_bounds
step = 500

atl_grid = gpd.GeoDataFrame(geometry = [
    shapely.geometry.box(minx, miny, maxx, maxy)
    for minx, maxx in zip(np.arange(a, c, step), np.arange(a, c, step)[1:])
    for miny, maxy in zip(np.arange(b, d, step), np.arange(b, d, step)[1:])], crs = atl_zone.crs)

# numeric id for each grid square to assist with aggregation
atl_grid['grid_id'] = range(1, len(atl_grid) + 1)

# spatial join grid and zoning map 
# this will create duplicate rows for any squares that contain two different zoning types
atl_grid = gpd.sjoin(atl_zone, atl_grid, how = 'inner', predicate = 'intersects')
atl_grid = atl_grid.drop(columns = 'index_right')

# reclassify grids by zoning type
atl_grid['area'] = atl_grid['geometry'].area
atl_grid['area_res'] = np.where(atl_grid['zoning'] == 'residential', atl_grid['area'], 0)
atl_grid['area_ind'] = np.where(atl_grid['zoning'] == 'industrial', atl_grid['area'], 0)

atl_grid_categ = atl_grid.groupby('grid_id').agg({'area': 'sum', 'area_res': 'sum', 'area_ind': 'sum'})
atl_grid_categ['pct_res'] = atl_grid_categ['area_res'] / atl_grid_categ['area']
atl_grid_categ['pct_ind'] = atl_grid_categ['area_ind'] / atl_grid_categ['area']

# classify based on relative percentages (may not be ideal)
atl_grid_categ['maj_zoning'] = np.where(atl_grid_categ['pct_res'] > atl_grid_categ['pct_ind'], 'residential', 'industrial')

# give each grid square its majority zoning
atl_grid = atl_grid.merge(atl_grid_categ[['maj_zoning']], left_on = 'grid_id', right_index = True)

# add in highway info
atl_hwy = gpd.sjoin(atl_grid, all_roads, how = 'left', predicate = 'intersects')

# make a dummy variable indicating the presence of a highway
atl_hwy['hwy'] = np.where(atl_hwy['designation'].isna(), 0, 1)

atl_hwy.to_file('data/output/atl_hwy.gpkg', driver = 'GPKG')