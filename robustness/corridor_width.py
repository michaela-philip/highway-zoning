import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample, merge_cnn_probs, add_cnn_interactions
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS,
    build_spec, leaveout_except,
)
from data_code.candidates import create_candidate_list

# robustness check: the direct/indirect split is defined by whether a grid square falls
# inside a corridor drawn between 1940 highway squares and the city's CBD (rays, optionally
# buffered into a wider corridor). Sweep that buffer width to see whether the CORE_VARS
# result found in the indirect ("outside corridor") sample is an artifact of how wide that
# corridor is drawn, rather than a discretionary-placement effect.
BUFFER_WIDTHS_M = [0, 100, 150, 200, 250, 300]
MIN_HWY_INDIRECT = 5
N_BOOTSTRAPS = 500


### FUNCTION TO FIT THE MAIN SPECIFICATION ON THE INDIRECT (OUTSIDE-CORRIDOR) SAMPLE ###
def fit_spec(df_indirect):
    x_vars, columns = build_spec(df_indirect, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS)
    table, beta, se, _ = bootstrap_lpm_table(df_indirect, x_vars, columns, n_bootstraps=N_BOOTSTRAPS)
    return table, beta, se, columns


df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)
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
        candidate_ids.update(create_candidate_list(city_data, centroids[city_mask], buffer_width_m=bw))

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

notes = "This table shows a collection of results testing the impact that construction of the highway corridor has on the estimated impact of Residential zoning and Majority Black status for grid squares outside of the highway corridor. " \
"The corridor is defined as a buffer around rays drawn between 1940 highway squares and the city's CBD, with the buffer width varied to test robustness." \
"Each column shows the results of a linear probability model estimated on the sample of grid squares outside the highway corridor, where the width of the highway corridor is varied. " \
"Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 500 draws. The model includes all controls included in the main analysis:variables related to housing, geographic, and demographic characteristics, as well as city fixed effects. " \
"The model also includes the logit values produced by the CNN and its interactions with the other variables." \

export_multiple_regressions(
    results,
    caption='Robustness to Highway-Candidate Corridor Width',
    label='tab:robustness/corridor_width',
    leaveout=leaveout_except(columns_ref, keep=[label for _, label in CORE_VARS]),
    notes = notes
)

print('\nsaved: tables/robustness/corridor_width.tex')
