import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy import stats

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except,
)

# robustness check: the direct/indirect split is defined by whether a grid square falls
# inside a corridor drawn between 1940 highway squares and the city's CBD (rays, optionally
# buffered into a wider corridor). Sweep that buffer width to see whether the CORE_VARS
# result found in the indirect ("outside corridor") sample is an artifact of how wide that
# corridor is drawn, rather than a discretionary-placement effect.
BUFFER_WIDTHS_M = [0, 150, 300, 500, 750, 1000, 1500, 2000]
MIN_HWY_INDIRECT = 5
N_BOOTSTRAPS = 500


### FUNCTION TO BUILD THE CANDIDATE (DIRECT) LIST AT A GIVEN CORRIDOR BUFFER WIDTH ###
### rays between each pair of 1940 highway squares, and between each 1940 highway square
### and the city's CBD, buffered into a corridor of the given width before intersecting
### the grid (buffer_width_m=0 replicates plain line intersection) ###
def candidate_list_at_width(city_data, city_cbd, buffer_width_m):
    pts = city_data.loc[city_data['hwy_40'] == 1]
    if pts.empty:
        return []

    centroids = pts.geometry.centroid.reset_index(drop=True)
    n = len(centroids)
    lines = [
        LineString([(centroids.iloc[i].x, centroids.iloc[i].y), (centroids.iloc[j].x, centroids.iloc[j].y)])
        for i, j in itertools.combinations(range(n), 2)
    ]
    cbd_point = city_cbd.geometry.iloc[0]
    lines += [LineString([(p.x, p.y), (cbd_point.x, cbd_point.y)]) for p in centroids]

    if buffer_width_m > 0:
        lines = [line.buffer(buffer_width_m) for line in lines]
    rays = gpd.GeoDataFrame(geometry=gpd.GeoSeries(lines, crs=city_data.crs))

    candidates = gpd.sjoin(city_data, rays, how='inner', predicate='intersects').drop_duplicates('grid_id')

    # restrict to squares near the city's typical elevation, and drop 1940 highway squares
    # themselves (they're not discretionary placements)
    elev_z = stats.zscore(candidates['dm_elevation'])
    candidates = candidates.loc[(elev_z > -1) & (elev_z < 1) & (candidates['hwy_40'] == 0)]
    return candidates['grid_id'].unique().tolist()


### FUNCTION TO FIT THE MAIN SPECIFICATION ON THE INDIRECT (OUTSIDE-CORRIDOR) SAMPLE ###
def fit_spec(df_indirect):
    x_vars, columns = build_spec(df_indirect, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)
    table, beta, se, _ = bootstrap_lpm_table(df_indirect, x_vars, columns, n_bootstraps=N_BOOTSTRAPS)
    return table, beta, se, columns


df = load_sample()
centroids = pd.read_csv('data/input/msas_with_central_city_cbds.csv')
centroids = gpd.GeoDataFrame(
    centroids,
    geometry=gpd.points_from_xy(centroids.cbd_retail_long, centroids.cbd_retail_lat),
    crs='EPSG:4267',  # best guess at CRS based off of projfinder.com
)

results = {}
columns_ref = None

for bw in BUFFER_WIDTHS_M:
    print(f"\nBuffer width: {bw}m")

    candidate_ids = set()
    for city in df['city'].unique():
        city_data = df.loc[df['city'] == city]
        city_mask = centroids['place'].str.lower().str.replace(' ', '') == city.lower().replace(' ', '')
        candidate_ids.update(candidate_list_at_width(city_data, centroids[city_mask], bw))

    df_direct = df.loc[df['grid_id'].isin(candidate_ids)]
    df_indirect = df.loc[~df['grid_id'].isin(candidate_ids)]

    pct_direct = len(df_direct) / len(df)
    n_hwy_indirect = int(df_indirect['hwy'].sum())
    print(f"  Direct: {len(df_direct):,} squares ({pct_direct:.1%}), {int(df_direct['hwy'].sum())} hwy")
    print(f"  Indirect: {len(df_indirect):,} squares, {n_hwy_indirect} hwy")

    if n_hwy_indirect < MIN_HWY_INDIRECT:
        print(f"  Skipping — fewer than {MIN_HWY_INDIRECT} highway squares in indirect sample")
        continue

    table, beta, se, columns = fit_spec(df_indirect)
    columns_ref = columns_ref or columns
    results[f'{bw}m'] = table

    print(f"  Residential x Black: {beta['Residential x Black']:+.4f} (SE={se['Residential x Black']:.4f})")

export_multiple_regressions(
    results,
    caption='Robustness to Highway-Candidate Corridor Width',
    label='tab:robustness/corridor_width',
    leaveout=leaveout_except(columns_ref, keep=[label for _, label in CORE_VARS]),
)

print('\nsaved: tables/robustness/corridor_width.tex')
