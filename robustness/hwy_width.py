import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from helpers.latex_formatting import export_multiple_regressions, format_regression_results
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except, fit_ols,
)

# grid squares are 150m (data_code/create_sample.py: gridsize=150). Thicken each highway
# centerline into a corridor BUFFER_SQUARES squares wide on either side of the line, i.e.
# BUFFER_M meters on each side -- motivated by the idea that a highway's effects (noise,
# pollution, severance) extend well past the grid square the line physically crosses.
GRIDSIZE_M = 150
BUFFER_SQUARES = 10
BUFFER_M = BUFFER_SQUARES * GRIDSIZE_M


### FUNCTION TO RECOMPUTE THE HWY INDICAR ###
### mirrors data_code/create_sample.py:place_highways, but buffers each highway
### centerline into a corridor before intersecting it with the grid ###
def widen_highways(grid):
    interstate = gpd.read_file('data/input/shapefiles/1960/interstates1959_del.shp').to_crs(grid.crs)
    state59 = gpd.read_file('data/input/shapefiles/1960/stateHighwayPaved1959_del.shp').to_crs(grid.crs)
    us59 = gpd.read_file('data/input/shapefiles/1960/usHighwayPaved1959_del.shp').to_crs(grid.crs)
    state40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/stateHighwayPaved1940_del.shp').to_crs(grid.crs)
    us40 = gpd.read_file('data/input/shapefiles/1940/1940 completed shapefiles/usHighwayPaved1940_del.shp').to_crs(grid.crs)

    # fclass information from Baik et al. (2010); eliminate all local roads
    interstate = interstate[~interstate['FCLASS'].isin([9, 19])]
    state59 = state59[~state59['FCLASS'].isin([9, 19])]
    us59 = us59[~us59['FCLASS'].isin([9, 19])]
    state40 = state40[~state40['FCLASS'].isin([9, 19])]
    us40 = us40[~us40['FCLASS'].isin([9, 19])]

    built_1959 = pd.concat([state59, us59, interstate])
    built_1940 = pd.concat([state40, us40])

    # thicken each highway centerline ine on either side
    corridor_1959 = gpd.GeoDataFrame(geometry=built_1959.buffer(BUFFER_M), crs=grid.crs)
    corridor_1940 = gpd.GeoDataFrame(geometry=built_1940.buffer(BUFFER_M), crs=grid.crs)

    grid_geo = grid[['grid_id', 'geometry']]

    hwy_59 = gpd.sjoin(grid_geo, corridor_1959, how='left', predicate='intersects')
    hwy_59['hwy_59_wide'] = np.where(hwy_59['index_right'].isna() 0, 1)
    hwy_59 = hwy_59.groupby('grid_id').agg({'hwy_59_wide': 'max'})

    hwy_40 = gpd.sjoin(grid_geo, corridor_1940, how='left', predicate='intersects')
    hwy_40['hwy_40_wide'] = np.where(hwy_40['index_right'].isna() 0, 1)
    hwy_40 = hwy_40.groupby('grid_id').agg({'hwy_40_wide':'max'})

    widened = hwy_59.join(hwy_40, how='outer').reset_index()
    widened['hwy_wide'] = (widened['hwy_59_wide'] - widened['hwy_40_wide']).clip(lower=0)
    return widened


### FUNCTION TO FIT THE MAIN SPECIFICATION ON A GIVEN SAMPLE ###
def fit_spec(df):
    df_restricted = restrict_to_discretionary(df)
    x_vars, columns = build_spec(df_restricted, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)
    results = format_regression_results(fit_ols(df_restricted, x_vars, columns))
    return results, columns


df = load_sample()

# recompute the hwy indicator on the widened corridor, city by city so each city's
# highways are reprojected/buffered/intersected in its own CRS (matches place_highways)
widened = pd.concat(
    [widen_highways(df.loc[df['city'] == city]) for city in df['city'].unique()],
    ignore_index=True,
)

df_wide = df.merge(widened, on='grid_id', how = 'left')

n_flip_59 = ((df_wide['hwy_59'] == 0) & (df_wide['hwy_59_wide'] == 1)).sum()
n_flip_40 = ((df_wide['hwy_40'] == 0) & (df_wide['hwy_40_wide'] == 1)).sum()
print(f'Widening corridor by {BUFFER_M}m ({BUFFER_SQUARES} grid squares of {GRIDSIZE_M}m) on either side of each highway centerline')
print(f'  hwy_59: {n_flip_59} additional grid squares now treated as having a highway')
print(f'  hwy_40: {n_flip_40} additional grid squares now treated as having a highway')

df_wide['hwy_40'] = df_wide['hwy_40_wide']
df_wide['hwy'] = df_wide['hwy_wide']
print(f'  hwy=1 squares: {int(df["hwy"].sum())} (baseline) -> {int(df_wide["hwy"].sum())} (widened)')

baseline_results, columns = fit_spec(df)
wide_results, _ = fit_spec(df_wide)

export_multiple_regressions(
    {"Baseline (Line Intersection)": baseline_results, f"Widened ({BUFFER_SQUARES} Squares Each Side)": wide_results},
    caption=f'Highway Width Robustness - ({BUFFER_SQUARES} Grid Squares Each Side)',
    label='tab:hwy_width_robustness',
    leaveout=leaveout_except(columns, keep=[label for _, label in CORE_VARS]),
)

print('\nsaved: tables/hwy_width_robustness.tex')