import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import geopandas as gpd

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results

df = pd.read_pickle('data/output/sample.pkl')
df['rent'] = df['rent'].replace(0, 0.00001)
df['valueh'] = df['valueh'].replace(0, 0.00001)

# prepare variables
df['log_valueh'] = np.log(df['valueh']) * df['valueh_avail']
df['log_rent'] = np.log(df['rent']) * df['rent_avail']
df['city_louisville'] = (df['city'] == 'louisville').astype(int)
df['city_littlerock'] = (df['city'] == 'littlerock').astype(int)
df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
df['log_dist_to_rr'] = np.log(df['dist_to_rr'])
df['log_dist_to_rr_sq'] = df['log_dist_to_rr'] ** 2
df['log_dist_to_hwy'] = np.log(df['dist_to_hwy'])
df['ResidentialxBlack'] = df['Residential'] * df['mblack_1945def']

# identify squares adjacent to existing highways
hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].copy()
all_squares = df[['grid_id', 'geometry']].copy()
touches_result = gpd.sjoin(
    all_squares,
    hwy_40_squares[['geometry']],
    how='left',
    predicate='touches'
)

# squares that got a match are adjacent to the 1940 network
adjacent_ids = set(
    touches_result[touches_result['index_right'].notna()]['grid_id']
)

df_restricted = df[~df['grid_id'].isin(adjacent_ids)].copy()
df_restricted = df[df['hwy_40'] == 0].copy() # drop any remaining 1940 highways that are not adjacent to others

x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'log_valueh', 'log_rent', 'log_dist_to_rr', 'log_dist_to_rr_sq', 'log_dist_to_hwy', 'distance_to_cbd', 'distance_to_cbd_sq', 'flood_risk', 'dist_water', 'slope', 'dm_elevation', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'dist(RR)', 'dist(RR^2)', 'ldist(Hwy)', 'dist(CBD)', 'dist(CBD^2)', 'Flood Risk', 'dist(Water)', 'Slope', 'Elevation', 'Owner', 'Number of Residents'] + [f'City_{c}' for c in df['city'].unique()[1:]]

# run and export regression
results_wholesample = format_regression_results(sm.OLS(df_restricted['hwy'], sm.add_constant(df_restricted[x_vars])).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['city']}))
export_single_regression(results_wholesample, caption = 'Determinants of Highway Placement - Full Sample', label = 'tab:wholesample_results', widthmultiplier=0.7, leaveout = ['log_valueh', 'log_rent', 'log_dist_to_rr', 'log_dist_to_rr_sq', 'log_dist_to_hwy', 'distance_to_cbd', 'distance_to_cbd_sq', 'flood_risk', 'dist_water', 'slope', 'dm_elevation', 'owner', 'numprec', 'city_louisville', 'city_littlerock'])