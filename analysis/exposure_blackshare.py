import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_characteristic_access, compute_shares
)
from analysis.lib.specs import (
    Spec, interact, cross, labels, leaveout_except, percentile_levels,
    RES, DEM_ACCESS, ANY_BLACK, SUIT, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, HWY_ACCESS,
)
from analysis.lib.marginal_effects import Comparison, Sweep, Strata, predicted_outcomes, export_predicted_outcomes_table
from analysis.lib.estimators import fit_logit_firth_conley
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_multiple_regressions, format_regression_results

cell_width = 150
df = load_sample(cell_width, impute = False)
df = merge_cnn_probs(df, f'predicted_activation-model5_{cell_width}*.csv', dataroot='cnn/')
df = df[df['zoning_change'] != 1]
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access', decay_m = 500, max_dist_m = 1000)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access_wide', decay_m = 1500, max_dist_m = 3000)

df_disc = restrict_to_discretionary(df)
df_disc = compute_characteristic_access(df_disc, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)

candidate_dict = get_candidate_dict(cell_width)
dir_sample, ind_sample = split_by_candidates(df_disc, candidate_dict)
SUIT_SWEEP = Sweep(SUIT, percentile_levels([10, 25, 50, 75, 90], where=lambda d: d['hwy'] == 1))
knot = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75)

ind_sample = compute_shares(ind_sample)
### Black == proximity to Black residents
ind_sample = compute_characteristic_access(ind_sample, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)
for city in ind_sample['city'].unique():
    city_mean = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].mean()
    city_std = ind_sample.loc[ind_sample['city'] == city, 'dem_access'].std()
    ind_sample.loc[ind_sample['city'] == city, 'dem_access'] = (ind_sample.loc[ind_sample['city'] == city, 'dem_access'] - city_mean)/ city_std

CORE = interact(RES, DEM_ACCESS)
LOGIT_INTERACTIONS = cross(CORE, SUIT.hinge(knot, 'High Suitability'))
BLACK_INTERACTION = [DEM_ACCESS * ANY_BLACK]
CONTROLS = [HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS]
PROTECTION = Comparison(effect=RES, across=DEM_ACCESS, effect_name='Protection Effect', difference_name='Disparate Protection')

spec1 = Spec(CORE, CNN_LOGIT, CONTROLS)
spec2 = Spec(CORE, CNN_LOGIT, LOGIT_INTERACTIONS, CONTROLS)
spec3 = Spec(CORE, CNN_LOGIT, BLACK_INTERACTION, CONTROLS)
spec4 = Spec(CORE, CNN_LOGIT, LOGIT_INTERACTIONS, BLACK_INTERACTION, CONTROLS)
fit1, fit2, fit3, fit4 = (fit_logit_firth_conley(ind_sample, spec, y_var='hwy', cutoff_m=1500) for spec in (spec1, spec2, spec3, spec4))

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/black_share}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values, including the indicator for the presence of any Black residents, " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit1, ind_sample, PROTECTION, columns=SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/black_share_sweep', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/black_share}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values, including the indicator for the presence of any Black residents, " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit2, ind_sample, PROTECTION, columns=SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/black_share_sweep_logitinteraction', notes = notes)

keep = labels(CORE, CNN_LOGIT, LOGIT_INTERACTIONS)
table1 = format_regression_results(fit1)
table2 = format_regression_results(fit2)
print(table1)
print(table2)
table = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_multiple_regressions(table, caption = 'Effect of Exposure to Black Residents on Highway Placement', label = 'tab:results/black_share', notes = notes, leaveout = leaveout_except(spec2.columns, keep=keep), widthmultiplier = 0.8)


keep = labels(CORE, CNN_LOGIT, LOGIT_INTERACTIONS, BLACK_INTERACTION)
table1 = format_regression_results(fit3)
table2 = format_regression_results(fit4)
print(table1)
print(table2)
table = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, demographic characteristics, and an indicator for the presence of Black residents. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Exposure to Black Residents is a location-specific weighted average of the share of Black homeowners in each square. Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. This measure is normalized within each city to account for city-level differences in Black homeownership and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_multiple_regressions(table, caption = 'Effect of Exposure to Black Residents on Highway Placement: Indicator for Black Residents', label = 'tab:results/black_share_indicator', notes = notes, leaveout = leaveout_except(spec4.columns, keep=keep), widthmultiplier = 1)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All other covariates are held constant at their own values and reported values are averaged across observations "\
"and can be interpreted as average marginal effects. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit3, ind_sample, PROTECTION)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction - Black Indicator', label = 'tab:predicted_outcomes/black_share_indicator', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit3, ind_sample, PROTECTION, columns=SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction - Suitability Sweep', label = 'tab:predicted_outcomes/black_share_indicator_sweep', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All other covariates are held constant at their own values and reported values are averaged across observations "\
"and can be interpreted as average marginal effects. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit4, ind_sample, PROTECTION)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction - Black Indicator and Linear Spline', label = 'tab:predicted_outcomes/black_share_indicator_logitinteraction', notes = notes)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(fit4, ind_sample, PROTECTION, columns=SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction - Black Indicator and Linear Spline', label = 'tab:predicted_outcomes/black_share_indicator_sweep_logitinteraction', notes = notes)

table = predicted_outcomes(fit4, ind_sample, PROTECTION,
                           columns=Strata(ANY_BLACK, [-0.5, 0.5, 1.5], labels=['No Black Residents', 'Has Black Residents']))
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/black_share_indicator}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents, and separately for squares with and without any Black residents of their own. All other covariates are held constant at their own values and reported values are averaged across observations, within each subset, and can be interpreted as average marginal effects. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction - Black Indicator and Linear Spline, by Presence of Black Residents', label = 'tab:predicted_outcomes/black_share_sweep_logitinteraction_anyblack_bystratum', notes = notes)
