import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# all shapefiles courtesy of Professors Taylor Jaworski and Carl Kitchens
interstates = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp')
state_paved = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp')
us_paved = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp')

all_roads = gpd.GeoDataFrame()

# fclass information from Baik et al. (2010)
# taking a subset designated as 'urban principal arterial'
interstates = interstates[(interstates['FCLASS'] == 11) | (interstates['FCLASS'] == 12) | (interstates['FCLASS'] == 14)]
state_paved = state_paved[(state_paved['FCLASS'] == 11) | (state_paved['FCLASS'] == 12) | (state_paved['FCLASS'] == 14)]
us_paved = us_paved[(us_paved['FCLASS'] == 11) | (us_paved['FCLASS'] == 12) | (us_paved['FCLASS'] == 14)]

# add designation column in case we want to know which original csv it came from
interstates['designation'] = 'interstate'
state_paved['designation'] = 'state_paved'
us_paved['designation'] = 'us_paved'

all_roads = pd.concat([interstates, state_paved, us_paved])
print('concat done', all_roads.shape, all_roads.columns)

all_roads.plot('designation')
plt.show()