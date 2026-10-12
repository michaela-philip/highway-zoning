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

spec1 = Spec(CORE, CONTROLS)
fit1 = fit_logit_firth_conley(ind_sample, spec1, y_var='hwy', cutoff_m=1500)
keep = labels(CORE)
results1 = format_regression_results(fit1)
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
"as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
"Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
"Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
"This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
" *** p < 0.01, ** p < 0.05, * p < 0.1"
export_single_regression(results1, caption = "Effect of Exposure to Black Residents on Likelihood of Highway Construction", 
                         label = 'tab:results/suitabilityxblackshare', leaveout = leaveout_except(spec1.columns, keep=keep),
                         notes=notes)

effect_levels = {'High Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75),
                 'Low Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.25)}
across_levels = {'White': ind_sample['dem_access'].quantile(0.25),
                 'Black': ind_sample['dem_access'].quantile(0.75)}
SUITABILITY = Comparison(effect = SUIT, effect_levels = effect_levels, across = DEM_ACCESS, across_levels = across_levels,
                         effect_name='Effect of Suitability', difference_name='Effect of Race')
outcomes1 = predicted_outcomes(fit1, ind_sample, SUITABILITY)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Table \\ref{tab:results/suitabilityxblackshare}, " \
"evaluated for areas differing in highway suitability and exposure to Black residents. High Suitability is defined as the 75th percentile of highway suitability" \
"and Low Suitability is defined as the 25th percentile of highway suitability. White areas are defined as those at the 25th percentile of " \
"Exposure to Black Residents, and Black areas are defined as those at the 75th percentile of Exposure to Black Residents. " \
"All other covariates are held constant at their own values. The Effect of Suitability is the difference in predicted rates for High and Low suitability areas." \
"The Effect of Race is the difference in the Effect of Suitability for White and Black areas. " \
"Standard errors are computed via the delta method from the spatial covariance matrix. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes1, caption = "Predicted Highway Construction by Exposure to Black Residents",
                         label = 'tab:predicted_outcomes/suitabilityxblackshare', notes = notes, widthmultiplier = 1)
plot_comparison(outcomes1, path = 'figures/suitabilityxblackshare.pdf')
export_figure_tex('figures/suitabilityxblackshare.pdf', caption = "Predicted Highway Construction by Exposure to Black Residents", 
                  label = 'fig:predicted_outcomes/suitabilityxblackshare_fig')

spec2 = Spec(CORE, PCT_RES, CONTROLS)
fit2 = fit_logit_firth_conley(ind_sample, spec2, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, PCT_RES)
results2 = format_regression_results(fit2)

CROSS = cross(CORE, PCT_RES)
spec3 = Spec(CORE, CROSS, PCT_RES, CONTROLS)
fit3 = fit_logit_firth_conley(ind_sample, spec3, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, CROSS, PCT_RES)
results3 = format_regression_results(fit3)

# export fit 2 and 3 together in one table
results2_3 = {'(1)': results2, '(2)': results3}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, " \
"as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. " \
"Highway Suitability is a logit value estimated for each square by a convolutional neural network based on geographical and geological factors. " \
"Exposure to Black Population is a location-specific weighted average of the share of Black residents in each square. " \
"Weights are computed by an exponential decay function of the straight-line distance between squares and set to zero beyond a distance of 3,000 meters. " \
"This measure is normalized within each city to account for city-level differences in Black population and concentration. " \
"Percent Residential is a measure of the proportion of the total area of each square that is zoned for residential use. Column 2 includes the interaction" \
"between Percent Residential and Highway Suitabilty and Exposure to Black Population." \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,500-meter distance cutoff. " \
" *** p < 0.01, ** p < 0.05, * p < 0.1"
export_multiple_regressions(results2_3, caption = "Effect of Exposure to Black Residents and Residential Zoning on Likelihood of Highway Construction",
                            label = 'tab:results/suitabilityxblackshare+xpctres', notes = notes, leaveout = leaveout_except(spec3.columns, keep = keep))

SWEEP = Sweep(PCT_RES, levels = {'0%': 0, '25%': 0.25, '50%': 0.5, '75%': 0.75, '100%': 1.0})
outcomes2 = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in column 1 of Table \\ref{tab:results/suitabilityxblackshare+xpctres}, " \
"evaluated for areas differing in highway suitability and exposure to Black residents. High Suitability is defined as the 75th percentile of highway suitability" \
"and Low Suitability is defined as the 25th percentile of highway suitability. White areas are defined as those at the 25th percentile of " \
"Exposure to Black Residents, and Black areas are defined as those at the 75th percentile of Exposure to Black Residents. " \
"All other covariates are held constant at their own values. The Effect of Suitability is the difference in predicted rates for High and Low suitability areas." \
"The Effect of Race is the difference in the Effect of Suitability for White and Black areas. " \
"Each column contains predicted outcomes and differences calculated for a specific level of Percent Residential. " \
"Standard errors are computed via the delta method from the spatial covariance matrix. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes2, caption = "Predicted Highway Construction Rates by Exposure to Black Residents and Residential Zoning",
                        label = 'tab:predicted_outcomes/suitabilityxblackshare+pctres', notes = notes, widthmultiplier = 1)
plot_effect_curves(fit2, ind_sample, SUITABILITY, sweep_var = PCT_RES, path = 'figures/predicted_outcomes_suitabilityxblackshare+pctres')
export_figure_tex('figures/predicted_outcomes_suitabilityxblackshare+pctres', label = 'fig:predicted_outcomes_suitabilityxblackshare+pctres',
                    caption = "Predicted Highway Construction Rates by Exposure to Black Residents and Residential Zoning")

outcomes3 = predicted_outcomes(fit3, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in column 2 of Table \\ref{tab:results/suitabilityxblackshare+xpctres}, " \
"evaluated for areas differing in highway suitability and exposure to Black residents. High Suitability is defined as the 75th percentile of highway suitability" \
"and Low Suitability is defined as the 25th percentile of highway suitability. White areas are defined as those at the 25th percentile of " \
"Exposure to Black Residents, and Black areas are defined as those at the 75th percentile of Exposure to Black Residents. " \
"All other covariates are held constant at their own values. The Effect of Suitability is the difference in predicted rates for High and Low suitability areas." \
"The Effect of Race is the difference in the Effect of Suitability for White and Black areas. " \
"Each column contains predicted outcomes and differences calculated for a specific level of Percent Residential. " \
"Standard errors are computed via the delta method from the spatial covariance matrix. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes3, caption = "Predicted Highway Construction Rates by Exposure to Black Residents and Residential Zoning",
                        label = 'tab:predicted_outcomes/suitabilityxblacksharexpctres', notes = notes, widthmultiplier = 1)

plot_predicted_curves(fit2, ind_sample, SUIT, sweep_var = DEM_ACCESS, effect_levels = effect_levels, path = 'figures/prob_curves_suitabilityxblackshare')
export_figure_tex('figures/prob_curves_suitabilityxblackshare3', label = 'fig:prob_curves_suitabilityxblackshare', 
                  caption = 'Predicted Highway Construction by Suitability and Exposure to Black Residents')