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
from analysis.lib.figures import plot_effect_curves, plot_comparison, export_figure_tex

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

spec2 = Spec(CORE, PCT_RES, CONTROLS)
fit2 = fit_logit_firth_conley(ind_sample, spec2, y_var='hwy', cutoff_m=1500)
keep = labels(CORE, PCT_RES)

suit_levels = {'High Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75),
                 'Low Suitability': ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.25)}
race_levels = {'Black': ind_sample['dem_access'].quantile(0.75), 
                 'White': ind_sample['dem_access'].quantile(0.25)}
SUITABILITY = Comparison(effect = SUIT, effect_levels = suit_levels, across = DEM_ACCESS, across_levels = race_levels,
                         effect_name='Effect of Suitability', difference_name='Racial Gap in Suitability Effect')
RACE = Comparison(effect = DEM_ACCESS, effect_levels = race_levels, across = SUIT, across_levels = suit_levels, 
                  effect_name='Difference in Highway Placement by Race', difference_name='Relative Effect of High Suitability on Black Neighborhoods')

# sweep across home values
SWEEP = Sweep(HOUSING_VARS[0], percentile_levels([10, 25, 50, 75, 90], where = lambda d: d['valueh_avail'] == 1))
outcomes1 = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in Table \\ref{tab:results/suitabilityxblackshare+pctres}. " \
"These probabilities are calculated by holding all covariates except for Highway Suitability, Exposure to Black Residents, and Log(Home Value) at their observed values. " \
"Twenty sets of counterfactuals are performed as follows:  High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; " \
"Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile; Log(Home Value) is set to the percentile value noted in each column. " \
"Predicted probabilities are averaged across observations for each counterfactual. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas. " \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes1, caption = "Predicted Probabilities: Heterogeneity by Home Value",
                        label = 'tab:heterogeneity/suitabilityxblacksharexvalueh', notes = notes, widthmultiplier = 1)

# sweep across black home values
SWEEP = Sweep(HOUSING_VARS[0], percentile_levels([10, 25, 50, 75, 90], where = lambda d: (d['valueh_avail'] == 1) & (d['black_homeowners_indicator'] == 1)))
outcomes1b = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)

# sweep across ownership levels
SWEEP = Sweep(HH_CONTROLS[0], {'0\\% Owner Occupied': 0, '25\\% Owner Occupied': 0.25, '50\\% Owner Occupied': 0.5, '75\\% Owner Occupied': 0.75, '100\\% Owner Occupied': 1.0})
outcomes2 = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in Table \\ref{tab:results/suitabilityxblackshare+pctres}. " \
"These probabilities are calculated by holding all covariates except for Highway Suitability, Exposure to Black Residents, " \
"and Percent Owner-Occupied at their observed values. " \
"Twenty sets of counterfactuals are performed as follows:  High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; " \
"Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile; Percent Owner-Occupied is set to the value noted in each column. " \
"Predicted probabilities are averaged across observations for each counterfactual. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas. " \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes2, caption = "Predicted Probabilites: Heterogeneity by Owner-Occupancy",
                        label = 'tab:heterogeneity/suitabilityxblacksharexowner', notes = notes, widthmultiplier = 1)

# sweep across number of residents
SWEEP = Sweep(HH_CONTROLS[1], percentile_levels([10, 25, 50, 75, 90]))
outcomes3 = predicted_outcomes(fit2, ind_sample, SUITABILITY, columns = SWEEP)
notes = "Average predicted probabilities of highway placement from the Firth's bias-reduced logistic regression in Table \\ref{tab:results/suitabilityxblackshare+pctres}. " \
"These probabilities are calculated by holding all covariates except for Highway Suitability, Exposure to Black Residents, " \
"and Log(Number of Residents) at their observed values. " \
"Twenty sets of counterfactuals are performed as follows:  High (Low) Suitability sets Highway Suitability to its 75th (25th) percentile; " \
"Black (White) areas set Exposure to Black Residents to its 75th (25th) percentile; Log(Number of Residents) is set to the percentile value noted in each column. " \
"Predicted probabilities are averaged across observations for each counterfactual. " \
"Effect of Suitability is the difference in predicted probability between High and Low Suitability areas. " \
"Racial Difference in Suitability Effect is the difference in the Effect of Suitability between Black and White areas. " \
"Standard errors are computed via the delta method from the spatial variance-covariance matrix and reported in parenthesis. " \
"P-values are only denoted for the differences reported in Panel B. *** p < 0.01, ** p < 0.05, * p < 0.1"
export_predicted_outcomes_table(outcomes3, caption = "Predicted Probabilities: Heterogeneity by Density",
                        label = 'tab:heterogeneity/suitabilityxblacksharexnumprec', notes = notes, widthmultiplier = 1) 