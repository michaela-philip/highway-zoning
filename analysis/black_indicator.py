import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_characteristic_access, widen_highways
)
from analysis.lib.specs import (
    Spec, interact, cross, labels, leaveout_except, percentile_levels,
    RES, MBLACK_MEAN_SHARE, SUIT, CNN_LOGIT, HOUSING_VARS, HH_RACE_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, HWY_ACCESS,
)
from analysis.lib.marginal_effects import Comparison, Sweep, predicted_outcomes, export_predicted_outcomes_table
from analysis.lib.estimators import fit_logit_firth_conley
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_multiple_regressions, format_regression_results

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
SUIT_SWEEP = Sweep(SUIT, percentile_levels([10, 25, 50, 75, 90], where=lambda d: d['hwy'] == 1))

ind_sample = widen_highways(ind_sample, buffer_m = 150)

knot = ind_sample.loc[ind_sample['hwy'] == 1, 'logit_normalized'].quantile(0.75)

print('black definition: mean share')
CORE = interact(RES, MBLACK_MEAN_SHARE)
LOGIT_INTERACTIONS = cross(CORE, SUIT.hinge(knot, 'High Suitability'))
CONTROLS = [HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS]
PROTECTION = Comparison(effect=RES, across=MBLACK_MEAN_SHARE, effect_name='Protection Effect', difference_name='Disparate Protection')

spec2 = Spec(CORE, CNN_LOGIT, LOGIT_INTERACTIONS, CONTROLS)
model = fit_logit_firth_conley(ind_sample, spec2, y_var = 'hwy', cutoff_m = 1500)
table2 = format_regression_results(model)
keep = labels(CORE, CNN_LOGIT, LOGIT_INTERACTIONS)

notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/mblack_mean_pct}, evaluated separately for Residential/Non-Residential areas and neighborhoods classified as Majority Black or White. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential for both Black and White neighborhoods. . *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(model, ind_sample, PROTECTION, columns = SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction with Linear Spline', label = 'tab:predicted_outcomes/mblack_mean_pct_logitinteraction', notes = notes)

spec1 = Spec(CORE, CNN_LOGIT, CONTROLS)
model = fit_logit_firth_conley(ind_sample, spec1, y_var = 'hwy', cutoff_m = 1500)
table1 = format_regression_results(model)
notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/mblack_mean_pct}, evaluated separately for Residential/Non-Residential areas and neighborhoods classified as Majority Black or White. All covariates other than Est. Highway Suitability are held constant at their own values " \
"while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential for both Black and White neighborhoods. . *** p < 0.01, ** p < 0.05, * p < 0.1"
table = predicted_outcomes(model, ind_sample, PROTECTION, columns = SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/mblack_mean_pct', notes = notes)


tables = {'(1)': table1, '(2)': table2}
notes = "Estimates from a Firth's bias-reduced logistic regression model of an indicator for highway construction from 1940-1959 on the covariates listed, as well as controls for housing, geographic, and demographic characteristics. The sample is restricted to squares that are outside the highway corridor. Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use. " \
"Black is an indicator for a square having a share of the city's Black population that is greater than the city's average share. " \
"Standard errors, reported in parenthesis, are \\textcite{conley_gmm_1999} spatial HAC standard errors with a 1,000-meter distance cutoff. Estimates in column (1) contain an uninteracted measure of highway suitability calculated by a convolutional neural network. Estimates in column (2) include the interaction between the coefficients of interest and a linear spline. *** p < 0.01, ** p < 0.05, * p < 0.1" 
export_multiple_regressions(tables, caption = 'Effect of Majority Black status on Highway Placement', label = 'tab:results/mblack_mean_pct', leaveout = leaveout_except(spec2.columns, keep = keep), notes = notes)
