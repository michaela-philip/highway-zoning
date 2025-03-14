import pandas as pd
import geopandas as gpd

ga = pd.read_csv('data/output/census_ga.csv')

# pull all observations in ATL - city code 0350
atl = ga[ga['city'] == 350]