import pandas as pd
import geopandas as gpd
import numpy as np

ed_gis = gpd.read_file('data/input/AtlantaGA40/AtlantaGA_ed40.shp')
ipums = pd.read_csv('data/output/ipums_ga.csv')

ipums['black'] = np.where(ipums['race'] == 2, 1, 0)
ipums['white'] = np.where(ipums['race'] == 1, 1, 0)

ipums_agg = ipums.groupby('enumdist')[['ownershp', 'rent', 'valueh', 'black', 'white', 'incwage']].mean()
ed_gis.rename(columns={'ed': 'enumdist'}, inplace=True)

ipums_ed = ipums_agg.merge(ed_gis, on = 'enumdist', how = 'left')

##now overlay (spatial join?) digitized zoning map and highway maps whenever they exist 