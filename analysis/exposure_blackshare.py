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
    RES, DEM_ACCESS, ANY_BLACK, SUIT, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, HWY_ACCESS, PCT_RES, MBLACK_MEAN_SHARE,
)
from analysis.lib.marginal_effects import Comparison, Sweep, Strata, predicted_outcomes, export_predicted_outcomes_table
from analysis.lib.estimators import fit_logit_firth_conley
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_multiple_regressions, format_regression_results, export_single_regression
from analysis.lib.figures import plot_effect_curves, plot_comparison, export_figure_tex, plot_predicted_curves

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

# suitability x Black exposure based on share
CORE = interact(DEM_ACCESS, SUIT)
CONTROLS = [HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS]

# define levels and comparison for predicted outcomes
effect_levels = {'High Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75),
                 'Low Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.25)}
across_levels = {'White': ind_sample['dem_access'].quantile(0.25),
                 'Black': ind_sample['dem_access'].quantile(0.75)}
SUITABILITY = Comparison(effect = SUIT, effect_levels = effect_levels, across = DEM_ACCESS, across_levels = across_levels,
                         effect_name='Effect of Suitability', difference_name='Racial Gap in Suitability Effect')

# run specifications
spec1 = Spec(CORE, CONTROLS)
fit1 = fit_logit_firth_conley(ind_sample, spec1, y_var='hwy', cutoff_m=1500)
keep = labels(CORE)
results1 = format_regression_results(fit1)
# notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
# "as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
# "Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
# "Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
# "Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
# "This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
# "Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
# " *** p < 0.01, ** p < 0.05, * p < 0.1"
# export_single_regression(results1, caption = "Effect of Exposure to Black Residents on Likelihood of Highway Construction", 
#                          label = 'tab:results/suitabilityxblackshare', leaveout = leaveout_except(spec1.columns, keep=keep),
#                          notes=notes)

# outcomes1 = predicted_outcomes(fit1, ind_sample, SUITABILITY)
# notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in " \
# "Table \\ref{tab:results/suitabilityxblackshare}. For each cell, Highway Suitability and Exposure to Black Residents are set to the values " \
# "below for every observation, all other covariates are held at their observed values, and predicted probabilities are averaged across observations. " \
# "High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile. " \
# "Effect of Suitability is the difference in predicted probability between High and Low Suitability areas." \
# "Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
# "Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
# "P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
# export_predicted_outcomes_table(outcomes1, caption = "Predicted Highway Construction by Exposure to Black Residents",
#                          label = 'tab:predicted_outcomes/suitabilityxblackshare', notes = notes, widthmultiplier = 1)

# spec 2 - add pct_res
spec2 = Spec(CORE, PCT_RES, CONTROLS)
fit2 = fit_logit_firth_conley(ind_sample, spec2, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, PCT_RES, HOUSING_VARS, HH_CONTROLS)
results2 = format_regression_results(fit2)

# export spec 1 and 2 together in one table
results1_2 = {'(1)': results1, '(2)': results2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
"as well as controls for geographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
"Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
"Percent Residential is a measure of the proportion of the total area of each square that is zoned for residential use. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
"Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
"This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
" *** p < 0.01, ** p < 0.05, * p < 0.1"
export_multiple_regressions(results1_2, caption = "Highway Suitability, Exposure to Black Residents, and Highway Placement", 
                            label = 'tab:results/suitabilityxblackshare+pctres', leaveout = leaveout_except(spec2.columns, keep=keep), notes=notes)

# export spec2 with all coefficients for appendix
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed. " \
"The sample is restricted to squares that are outside the highway corridor. " \
"Distance-Decayed Highway Access is a measure of the accessibility of each square to the existing (pre-1940) highway network, " \
"weighted by the exponential decay function, $\\e^(-\\tau / \\sigma)$, where $\\tau$ is the straight-line distance between squares, $\\sigma = 500$, " \
"and the weight is set to 0 beyond a distance of 1,000 meters. Distance-Decayed Highway Access (Wide) is the same measure calculated where $\\sigma = 1500$ and " \
"the weight is set to 0 beyond a distance of 3,000 meters. " \
"Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
"Percent Residential is a measure of the proportion of the total area of each square that is zoned for residential use. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
"Weights are computed by the exponential decay function, $\\e^(-\\tau / \\sigma)$, where $\\tau$ isthe straight-line distance between squares, " \
"$\\sigma$ is the decay parameter set to 600, and weights are set to zero beyond a distance of 3,000 meters. " \
"This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
" *** p < 0.01, ** p < 0.05, * p < 0.1"
export_single_regression(results2, caption = "All Coefficients from Primary Specification", label = 'tab:results/suitabilityxblackshare+pctres_full')

# predicted probabilities for spec 2 no sweep
outcomes1 = predicted_outcomes(fit2, ind_sample, SUITABILITY)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in " \
"Table \\ref{tab:results/suitabilityxblackshare}. For each cell, Highway Suitability and Exposure to Black Residents are set to the values " \
"below for every observation, all other covariates are held at their observed values, and predicted probabilities are averaged across observations. " \
"High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas." \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes1, caption = "Predicted Probabilities: Exposure to Black Residents and Highway Suitability",
                         label = 'tab:predicted_outcomes/suitabilityxblackshare', notes = notes, widthmultiplier = 1)

# predicted probabilities for spec 2 with sweep
SWEEP = Sweep(PCT_RES, levels = {'0//%': 0, '25//%': 0.25, '50//%': 0.5, '75//%': 0.75, '100//%': 1.0})
outcomes2 = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in column 2 of " \
"Table \\ref{tab:results/suitabilityxblackshare+pctres}. " \
"For each cell, Highway Suitability and Exposure to Black Residents are set to the values below for every observation, " \
"Percent Residential is set to the value noted in the column header, and all other covariates are held at their observed values. " \
"Predicted probabilities are averaged across observations. " \
"High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas." \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes2, caption = "Predicted Probabilities: Exposure to Black Residents and Highway Suitability, by Percent Residential",
                        label = 'tab:predicted_outcomes/suitabilityxblackshare+pctres', notes = notes, widthmultiplier = 1)
# plot_effect_curves(fit2, ind_sample, SUITABILITY, sweep_var = PCT_RES, path = 'figures/predicted_outcomes_suitabilityxblackshare+pctres.png')
# export_figure_tex('figures/predicted_outcomes_suitabilityxblackshare+pctres.png', label = 'fig:predicted_outcomes_suitabilityxblackshare+pctres',
#                     caption = "Predicted Probabilities: Exposure to Black Residents and Highway Suitability, by Percent Residential")

# spec 3 - interaction with pct_res
CROSS = cross(CORE, PCT_RES)
spec3 = Spec(CORE, CROSS, PCT_RES, CONTROLS)
fit3 = fit_logit_firth_conley(ind_sample, spec3, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, CROSS, PCT_RES)
results3 = format_regression_results(fit3)

# export fit 2 and 3 together in one table
# results2_3 = {'(1)': results2, '(2)': results3}
# notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
# "as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
# "Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
# "Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
# "Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
# "This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
# "Percent Residential is a measure of the proportion of the total area of each square that is zoned for residential use. Column 2 includes the interaction" \
# "between Percent Residential and Highway Suitabilty and Exposure to Black Population." \
# "Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
# " *** p < 0.01, ** p < 0.05, * p < 0.1"
# export_multiple_regressions(results2_3, caption = "Effect of Exposure to Black Residents and Residential Zoning on Likelihood of Highway Construction",
#                             label = 'tab:results/suitabilityxblackshare+xpctres', notes = notes, leaveout = leaveout_except(spec3.columns, keep = keep))

# export fit 3 alone
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
"as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
"Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
"Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
"This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Percent Residential is a measure of the proportion of the total area of each square that is zoned for residential use. This model also includes interactions " \
"between Percent Residential, Highway Suitability, and Exposure to Black Residents. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
" *** p < 0.01, ** p < 0.05, * p < 0.1"
export_single_regression(results3, caption = "Highway Suitability, Exposure to Black Residents, and Highway Placement: Interaction with Residential Zoning",
                         label = 'tab:results/suitabilityxblacksharexpctres', leaveout = leaveout_except(spec3.columns, keep=keep), notes=notes)

# predicted probabilities for spec 3 with sweep
outcomes3 = predicted_outcomes(fit3, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in " \
"Table \\ref{tab:results/suitabilityxblacksharexpctres}. " \
"For each cell, Highway Suitability and Exposure to Black Residents are set to the values below for every observation, " \
"Percent Residential is set to the value noted in the column header, and all other covariates are held at their observed values. " \
"Predicted probabilities are averaged across observations. " \
"High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas." \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes3, caption = "Predicted Probabilities: Triple Interaction, by Percent Residential",
                        label = 'tab:predicted_outcomes/suitabilityxblacksharexpctres', notes = notes, widthmultiplier = 1)

plot_predicted_curves(fit2, ind_sample, SUIT, sweep_var = DEM_ACCESS, effect_levels = effect_levels, path = 'figures/prob_curves_suitabilityxblackshare')
export_figure_tex('figures/prob_curves_suitabilityxblackshare', label = 'fig:prob_curves_suitabilityxblackshare', 
                  caption = 'Highway Likelihood by Exposure to Black Residents: High Suitability vs. Low Suitability Areas')

effect_levels = {'Black': ind_sample['dem_access'].quantile(0.75),
                  'White': ind_sample['dem_access'].quantile(0.25)}
plot_predicted_curves(fit2, ind_sample, effect = DEM_ACCESS, sweep_var = SUIT, effect_levels = effect_levels, path = 'figures/prob_curves_blacksharexsuitability')
export_figure_tex('figures/prob_curves_blacksharexsuitability', label = 'fig:prob_curves_blacksharexsuitability', 
                  caption = 'Highway Likelihood by Highway Suitability: Black vs. White Areas')