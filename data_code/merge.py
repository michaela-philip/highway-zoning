import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely.geometry
# from arcgis import GIS
# from arcgis.features.analysis import overlay_layers

################ AGGREGATE HIGHWAY DATA ################
# all shapefiles courtesy of Professors Taylor Jaworski and Carl Kitchens
interstates = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp')
state_paved_59 = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us_paved_59 = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')
state_paved_40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp')
us_paved_40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp')

# change crs to ESPG 3857
interstates = interstates.to_crs('EPSG:3857')
state_paved_59 = state_paved_59.to_crs('EPSG:3857')
us_paved_59 = us_paved_59.to_crs('EPSG:3857')
state_paved_40 = state_paved_40.to_crs('EPSG:3857')
us_paved_40 = us_paved_40.to_crs('EPSG:3857')

# fclass information from Baik et al. (2010)
# eliminating all local roads
interstates = interstates[~interstates['FCLASS'].isin([9, 19])]
state_paved_59 = state_paved_59[~state_paved_59['FCLASS'].isin([9,19])]
us_paved_59 = us_paved_59[~us_paved_59['FCLASS'].isin([9,19])]
state_paved_40 = state_paved_40[~state_paved_40['FCLASS'].isin([9,19])]
us_paved_40 = us_paved_40[~us_paved_40['FCLASS'].isin([9,19])]

# expand the lines to ensure that we are correctly capturing overlap
# us_paved_40['geometry'] = us_paved_40.buffer(2000)
# us_paved_59['geometry'] = us_paved_59.buffer(2000)
# state_paved_40['geometry'] = state_paved_40.buffer(5)
# state_paved_59['geometry'] = state_paved_59.buffer(5)

# include only roads that exist in 1959 and did not exist in 1940
state_paved = state_paved_59.overlay(state_paved_40, how = 'difference')
state_paved = state_paved[~state_paved.is_empty]
us_paved = us_paved_59.overlay(us_paved_40, how = 'difference', keep_geom_type = False)
us_paved = us_paved[~us_paved.is_empty]

# # use overlay to remove roads built before 1940
# portal = GIS(username='mphili6_emory', password='FLY3@gl3$22!')
# state_paved = overlay_layers(input_layer = state_paved_40, overlay_layer = state_paved_59, overlay_type = 'erase')

# combine all roads
all_roads = pd.concat([interstates, state_paved, us_paved])


################ CREATE THE GRID ################
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


################ ADD ZONING INFO TO GRID ################
# try this as an overlay instead
atl_grid_zone = atl_zone.overlay(atl_grid, how = 'identity')

# reclassify grids by zoning type (similar approach Noelkel et al. (2020) HOLC classification)
atl_grid_zone['area'] = atl_grid_zone['geometry'].area
atl_grid_zone['area_res'] = np.where(atl_grid_zone['zoning'] == 'residential', atl_grid_zone['area'], 0)
atl_grid_zone['area_ind'] = np.where(atl_grid_zone['zoning'] == 'industrial', atl_grid_zone['area'], 0)
atl_grid_zone = atl_grid_zone.groupby('grid_id').agg({'area': 'sum', 'area_res': 'sum', 'area_ind': 'sum'})
atl_grid_zone['pct_res'] = atl_grid_zone['area_res'] / atl_grid_zone['area']
atl_grid_zone['pct_ind'] = atl_grid_zone['area_ind'] / atl_grid_zone['area']

# classify based on relative percentages (may not be ideal)
atl_grid_zone['maj_zoning'] = np.where(atl_grid_zone['pct_res'] > atl_grid_zone['pct_ind'], 'residential', 'industrial')

# give each grid square its majority zoning
atl_grid = atl_grid.merge(atl_grid_zone[['maj_zoning']], left_on = 'grid_id', right_index = True)

# save this separately
atl_grid.to_file('data/output/atl_grid.gpkg', driver = 'GPKG')


################ ADD HIGHWAY INFO TO GRID ################
atl_grid_hwy = gpd.sjoin(atl_grid, all_roads, how = 'left', predicate = 'intersects')

# make a dummy variable indicating the presence of a highway
atl_grid_hwy['hwy'] = np.where(atl_grid_hwy['type'].isna(), 0, 1)

# aggregate by grid_id taking max value (if any highways exist, it is 1 no matter what)
atl_grid_hwy = atl_grid_hwy.groupby('grid_id').agg({'hwy': 'max'})

# merge in hwy indicator
atl_hwy = atl_grid.merge(atl_grid_hwy, left_on = 'grid_id', right_index = True)

atl_hwy.to_file('data/output/atl_hwy.gpkg', driver = 'GPKG')
print('done')