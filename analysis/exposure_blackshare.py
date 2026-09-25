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
from analysis.lib.estimators import (fit_ppml_conley, fit_ppml_firth, fit_ppml_firth_conley, fit_logit_firth_conley)
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
df_disc = compute_characteristic_access(df_disc, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)

candidate_dict = get_candidate_dict(cell_width)
dir_sample, ind_sample = split_by_candidates(df_disc, candidate_dict)
sweep_values = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
sweep_cols = {sweep_values[0]: '10th Percentile',
                sweep_values[1] : '25th Percentile',
                sweep_values[2] : '50th Percentile',
                sweep_values[3] : '75th Percentile',
                sweep_values[4] : '90th Percentile'}

knot = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75)
ind_sample['logit_above_knot'] = np.maximum(0, ind_sample['logit_normalized'] - knot)

ind_sample = compute_shares(ind_sample)
### Black == proximity to Black residents
ind_sample = compute_characteristic_access(ind_sample, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)
for city in ind_sample['city'].unique():
    city_mean = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].mean()
    city_std = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].std()
    ind_sample.loc[ind_sample['city'] == city, 'dem_access'] = (ind_sample.loc[ind_sample['city'] == city, 'dem_access'] - city_mean)/ city_std

CORE = core_spec(ind_sample, 'dem_access')
LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, 'dem_access', 'logit_above_knot', 'High Suitability')
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS]
table1 = format_regression_results(model)
print(table1)
table1 = table1.rename(index={
    'Black': 'Exposure to Black Population',
    'Residential x Black': 'Residential x Exposure to Black Population',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x Est. Highway Suitability',
    'Black x Logit': 'Exposure to Black Population x Est. Highway Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Population x Est. Highway Suitability',
})
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS]
table2 = format_regression_results(model)
print(table2)
table2 = table2.rename(index={
    'Black': 'Exposure to Black Population',
    'Residential x Black': 'Residential x Exposure to Black Population',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x Est. Highway Suitability',
    'Black x Logit': 'Exposure to Black Population x Est. Highway Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Population x Est. Highway Suitability',
})

table = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1" 
export_multiple_regressions(table, caption = 'Effect of Exposure to Black REsidents on Highway Placement', label = 'tab:outside_corridor_black_share', notes = notes, leaveout = leaveout_except(columns, keep=keep), widthmultiplier = 0.8)

BLACK_INTERACTION = sweep_interactions_spec(ind_sample, 'dem_access', 'any_black', 'Any Black Residents')
BLACK_INTERACTION = [BLACK_INTERACTION[1]]
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, BLACK_INTERACTION, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS + BLACK_INTERACTION]
table1 = format_regression_results(model)
print(table1)
table1 = table1.rename(index={
    'Black': 'Exposure to Black Residents',
    'Residential x Black': 'Residential x Exposure to Black Residents',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x Est. Highway Suitability',
    'Black x Logit': 'Exposure to Black Residents x Est. Highway Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Residents x Est. Highway Suitability',
})

x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, BLACK_INTERACTION, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
table2 = format_regression_results(model)
print(table2)
table2 = table2.rename(index={
    'Black': 'Exposure to Black Residents',
    'Residential x Black': 'Residential x Exposure to Black Residents',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x Est. Highway Suitability',
    'Black x Logit': 'Exposure to Black Residents x Est. Highway Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Residents x Est. Highway Suitability',
})
table = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, demographic characteristics, and an indicator for the presence of Black residents. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Exposure to Black Residents is a location-specific weighted average of the share of Black homeowners in each square. Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences in Black homeownership and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1" 
export_multiple_regressions(table, caption = 'Effect of Exposure to Black Residents on Highway Placement: Indicator for Black Residents', label = 'tab:outside_corridor_black_share_indicator', notes = notes, leaveout = leaveout_except(columns, keep=keep), widthmultiplier = 0.8)

x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:outside_corridor_black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All other covariates are held constant at their own values and reported values are averaged across observations "\
"and can be interpreted as average marginal effects. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1" 
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'])
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction by Zoning Designation and Exposure to Black Residents', label = 'tab:predicted_outcomes/black_share', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:outside_corridor_black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1" 
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], 
                                    sweep_interactions = [], sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction by Zoning Designation and Exposure to Black Residents: Suitability Sweep', label = 'tab:predicted_outcomes/black_share_sweep', notes = notes)

x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:outside_corridor_black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All other covariates are held constant at their own values and reported values are averaged across observations "\
"and can be interpreted as average marginal effects. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1" 
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'])
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction by Zoning Designation and Exposure to Black Residents with Linear Spline', label = 'tab:predicted_outcomes/black_share_logitinteraction', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:outside_corridor_black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1" 
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], 
                                    sweep_interactions = LOGIT_INTERACTIONS, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction by Zoning Designation and Exposure to Black Residents with Linear Spline: Suitability Sweep', label = 'tab:predicted_outcomes/black_share_sweep_logitinteraction', notes = notes)



