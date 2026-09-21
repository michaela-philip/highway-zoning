import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_characteristic_access, assign_highway_exposure, widen_highways, city_specific_shares, compute_shares, high_access_indicator
)
from analysis.lib.bootstrap import bootstrap_lpm_table, fit_ols
from analysis.lib.specs import (
    CONTINUOUS_EXPOSURE, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, CNN_LOGIT, RESIDENTIAL, HWY_ACCESS,
    build_spec, leaveout_except, core_spec, sweep_interactions_spec, black_definition_note,
)
from analysis.lib.marginal_effects import (
    export_predicted_outcomes_table, predicted_outcomes_from_fit, predicted_outcomes_by_stratum_from_fit, export_predicted_outcomes_table
)
from analysis.lib.estimators import (fit_ppml_conley, fit_ppml_firth, fit_ppml_firth_conley)
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, export_single_regression

cell_width = 150
df = load_sample(cell_width, impute = False)
df = merge_cnn_probs(df, f'predicted_activation-model1_{cell_width}*.csv', dataroot='cnn/')
df = df[df['zoning_change'] != 1]
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access', decay_m = 500, max_dist_m = 1000)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access_wide', decay_m = 1500, max_dist_m = 3000)

df_disc = restrict_to_discretionary(df)

candidate_dict = get_candidate_dict(cell_width)
dir_sample, ind_sample = split_by_candidates(df_disc, candidate_dict)
sweep_values = ind_sample['logit_normalized'].quantile([0.25, 0.50, 0.75, 0.90]).tolist()

print('homeowners access - falsification')
ind_sample = compute_shares(ind_sample)
ind_sample = compute_characteristic_access(ind_sample, 'share_homeowners', 'homeowners_access', decay_m = 300, max_dist_m = 3000)
for city in ind_sample['city'].unique():
    city_mean = ind_sample.loc[ind_sample['city'] == city, 'homeowners_access'].mean()
    city_std = ind_sample.loc[ind_sample['city'] == city, 'homeowners_access'].std()
    ind_sample.loc[ind_sample['city'] == city, 'homeowners_access'] = (ind_sample.loc[ind_sample['city'] == city, 'homeowners_access'] - city_mean)/ city_std
CORE = core_spec(ind_sample, 'homeowners_access')
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_ppml_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1000)
table = format_regression_results(model)

keep = [label for _, label in CORE + CNN_LOGIT]
notes = "Estimates from a Poisson pseudo-maximum likelihood model (PPML) regression of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether the grid square is zoned for residential use. " \
"Exposure to Homeowners is a location-specific weighted average of the share of homeowners in each square. Weights are exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. *** p < 0.01, ** p < 0.05, * p < 0.1" 
table = table.rename(index={
    'Black': 'Exposure to Homeowners',
    'Residential x Black': 'Residential x Exposure to Homeowners',
    'Black x CNN Logit': 'Exposure to Homeowners x CNN Logit',
    'Residential x Black x CNN Logit': 'Residential x Exposure to Homeowners x CNN Logit',
})
export_single_regression(table, caption = 'Falsification Test: Effect of Exposure to Homeowners on Highway Placement', label = 'tab:robustness/falsification_homeowners', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

LO, HI = ind_sample['homeowners_access'].quantile([0.25, 0.85])
predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'log', eval_at = 'ame', black_values = [LO, HI], black_labels = ['Low Homeowner Access', 'High Homeowner Access'], sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values, sweep_interactions = [])