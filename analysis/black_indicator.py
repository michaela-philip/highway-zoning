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
    CONTINUOUS_EXPOSURE, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, CNN_LOGIT, RESIDENTIAL, HWY_ACCESS, HH_RACE_CONTROLS, HWY_40,
    build_spec, leaveout_except, core_spec, sweep_interactions_spec, black_definition_note,
)
from analysis.lib.marginal_effects import (
    export_predicted_outcomes_table, predicted_outcomes_from_fit, predicted_outcomes_by_stratum_from_fit, export_predicted_outcomes_table
)
from analysis.lib.estimators import (fit_ppml_conley, fit_ppml_firth, fit_ppml_firth_conley, fit_logit_firth, fit_logit_firth_conley)
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, export_single_regression

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
ind_sample = ind_sample[ind_sample['numprec'] > 10]
sweep_values = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
sweep_bins = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
sweep_cols = {sweep_values[0]: '10th Percentile',
                sweep_values[1] : '25th Percentile',
                sweep_values[2] : '50th Percentile',
                sweep_values[3] : '75th Percentile',
                sweep_values[4] : '90th Percentile'}

ind_sample = widen_highways(ind_sample, buffer_m = 150)

knot = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75)
ind_sample['logit_above_knot'] = np.maximum(0, ind_sample['logit_normalized'] - knot)

print('black definition: mean share')
CORE = core_spec(ind_sample, 'mean_share')
LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, 'mean_share', 'logit_above_knot', 'High Suitability')
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 1500)
table2 = format_regression_results(model)
keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS]

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/mblack_mean_pct}, evaluated separately for Residential/Non-Residential areas and neighborhoods classified as Majority Black or White. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential for both Black and White neighborhoods. . *** p < 0.01, ** p < 0.05, * p < 0.1"
# predicted_outcomes_by_stratum_from_fit(model, ind_sample, x_vars, columns, link = 'logit', sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS, bins = sweep_bins)
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction with Linear Spline', label = 'tab:predicted_outcomes/mblack_mean_pct_logitinteraction', notes = notes)

x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 1500)
table1 = format_regression_results(model)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/mblack_mean_pct}, evaluated separately for Residential/Non-Residential areas and neighborhoods classified as Majority Black or White. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential for both Black and White neighborhoods. . *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = [])
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/mblack_mean_pct', notes = notes)


tables = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Black is an indicator for a square having a share of the city's Black population that is greater than the city's average share. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1" 
export_multiple_regressions(tables, caption = 'Effect of Majority Black status on Highway Placement', label = 'tab:results/mblack_mean_pct', leaveout = leaveout_except(columns, keep = keep), notes = notes)
# print('black definition: mean pct')
# CORE = core_spec(ind_sample, 'mean_pct')
# LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, 'mean_pct', 'logit_above_knot', 'High Suitability')
# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
# model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 2000)
# print(format_regression_results(model))
# predicted_outcomes_by_stratum_from_fit(model, ind_sample, x_vars, columns, link = 'logit', sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS, bins = sweep_bins)
# predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS)

# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
# model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 2000)
# print(format_regression_results(model))
# predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = [])

# CORE = core_spec(ind_sample, 'mean_share')
# LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, 'mean_share', 'logit_normalized', 'Est. Highway Suitability')
# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
# model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 2000)
# print(format_regression_results(model))
# predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'mean', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = [])