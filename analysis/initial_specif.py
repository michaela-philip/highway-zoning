import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_characteristic_access, compute_shares
)
from analysis.lib.specs import (
    Spec, labels, leaveout_except,
    RES, DEM_ACCESS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, HWY_ACCESS,
)
from analysis.lib.estimators import fit_logit_firth_conley
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_single_regression, format_regression_results

cell_width = 150
df = load_sample(cell_width, impute = False)
df = merge_cnn_probs(df, f'predicted_activation-model5_{cell_width}*.csv', dataroot='cnn/')
df = df[df['zoning_change'] != 1]
df['Residential'] = np.where(df['pct_res'] >= 0.75, 1, 0)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access', decay_m = 600, max_dist_m = 1500)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access_wide', decay_m = 1500, max_dist_m = 5000)

df_disc = restrict_to_discretionary(df)

candidate_dict = get_candidate_dict(cell_width)
dir_sample, ind_sample = split_by_candidates(df_disc, candidate_dict)

ind_sample = compute_shares(ind_sample)
ind_sample = compute_characteristic_access(ind_sample, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)
for city in ind_sample['city'].unique():
    city_mean = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].mean()
    city_std = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].std()
    ind_sample.loc[ind_sample['city'] == city, 'dem_access'] = (ind_sample.loc[ind_sample['city'] == city, 'dem_access'] - city_mean)/ city_std

# no Residential x Exposure interaction in this specification
CORE = [RES, DEM_ACCESS]
spec = Spec(CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS)
model = fit_logit_firth_conley(ind_sample, spec, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, CNN_LOGIT)
table1 = format_regression_results(model)
print(table1)
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Exposure to Black Homeowners is a location-specific weighted average of the share of Black homeowners in each square. Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff.  *** p < 0.01, ** p < 0.05, * p < 0.1"
export_single_regression(table1, caption = 'Effect of Zoning and Exposure to Black Residents on Highway Placement', label = 'tab:results/uninteracted_spec', notes = notes, leaveout = leaveout_except(spec.columns, keep=keep), widthmultiplier = 0.6)
