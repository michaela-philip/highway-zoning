import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except,
)

# grid squares are 150m (data_code/create_sample.py: gridsize=150). Buffer each hwy==1
# square into a corridor BUFFER_SQUARES squares wide on either side, i.e. BUFFER_M meters
# -- motivated by the idea that a highway's effects (noise, pollution, severance) extend
# well past the grid square the line physically crosses.
GRIDSIZE_M = 150
BUFFER_SQUARES = 10
BUFFER_M = BUFFER_SQUARES * GRIDSIZE_M


### FUNCTION TO WIDEN THE HWY INDICATOR ###
### buffers the geometry of squares already flagged hwy==1 (i.e. the discretionary,
### newly-built-by-1959 squares -- see data_code/create_sample.py:452, hwy = clip(hwy_59 - hwy_40, 0))
### into a corridor, then marks any grid square intersecting that corridor as widened ###
def widen_highways(grid):
    hwy_squares = grid.loc[grid['hwy'] == 1, ['grid_id', 'geometry']]
    corridor = gpd.GeoDataFrame(geometry=hwy_squares.buffer(BUFFER_M), crs=grid.crs)

    grid_geo = grid[['grid_id', 'geometry']]
    hwy_wide = gpd.sjoin(grid_geo, corridor, how='left', predicate='intersects')
    hwy_wide['hwy_wide'] = np.where(hwy_wide['index_right'].isna(), 0, 1)
    hwy_wide = hwy_wide.groupby('grid_id').agg({'hwy_wide': 'max'}).reset_index()
    return hwy_wide


### FUNCTION TO FIT THE MAIN SPECIFICATION ON A GIVEN SAMPLE ###
def fit_spec(df):
    df_restricted = restrict_to_discretionary(df)
    x_vars, columns = build_spec(df_restricted, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)
    results, *_ = bootstrap_lpm_table(df_restricted, x_vars, columns)
    return results, columns


df = load_sample()

# widen the hwy indicator, city by city so each city's squares are buffered/intersected
# in its own CRS
widened = pd.concat(
    [widen_highways(df.loc[df['city'] == city]) for city in df['city'].unique()],
    ignore_index=True,
)

df_wide = df.merge(widened, on='grid_id', how = 'left')

n_flip = ((df_wide['hwy'] == 0) & (df_wide['hwy_wide'] == 1)).sum()
print(f'Widening corridor by {BUFFER_M}m ({BUFFER_SQUARES} grid squares of {GRIDSIZE_M}m) on either side of each hwy=1 square')
print(f'  {n_flip} additional grid squares now treated as having a highway')

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