import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
from analysis.lib.figures import plot_access_map, export_figure_tex


cell_width = 150
df = load_sample(cell_width, impute = False)
df = merge_cnn_probs(df, f'predicted_activation-model5_{cell_width}*.csv', dataroot='cnn/')
df = df[df['zoning_change'] != 1]
df['Residential'] = np.where(df['pct_res'] >= 0.75, 1, 0)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access', decay_m = 600, max_dist_m = 1500)
df = compute_characteristic_access(df, 'hwy_40', 'hwy_access_wide', decay_m = 1500, max_dist_m = 5000)


# map of estimated highway suitability
city_grid = df[df['city'] == 'atlanta']
plot_access_map(df[df['city'] == 'atlanta'], city_grid, 'logit_normalized',
                path = f'figures/logit_normalized_atlanta.pdf',
                colorbar_label = 'Estimated Highway Suitability (normalized)', hwy_40 = False)

df_disc = restrict_to_discretionary(df)
print(len(df) - len(df_disc), 'squares dropped')
df_disc = compute_characteristic_access(df_disc, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)

candidate_dict = get_candidate_dict(cell_width)
dir_sample, ind_sample = split_by_candidates(df_disc, candidate_dict)
print(len(df_disc) - len(ind_sample), 'squares dropped from individual sample')

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
keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS]
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
table = format_regression_results(model)
table = table.rename(index = {
    'Black': 'Exposure to Black Population',
    'Residential x Black': 'Residential x Exposure to Black Population',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x High Suitability',
    'Black x Logit': 'Exposure to Black Population x High Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Population x High Suitability',
})
export_single_regression(table, caption = 'Exposure to Black Population', label = 'tab:slides/results/black_share_nointeraction', leaveout = leaveout_except(columns, keep = keep))
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], sweep_interactions = [], sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:slides/predicted_outcomes/black_share_sweep', column_labels = sweep_cols)
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'])

x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=True)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
table = format_regression_results(model)
table = table.rename(index = {
    'Black': 'Exposure to Black Population',
    'Residential x Black': 'Residential x Exposure to Black Population',
    'Logit': 'Est. Highway Suitability',
    'Residential x Logit': 'Residential x High Suitability',
    'Black x Logit': 'Exposure to Black Population x High Suitability',
    'Residential x Black x Logit': 'Residential x Exposure to Black Population x High Suitability',
})
export_single_regression(table, caption = 'Exposure to Black Population', label = 'tab:slides/results/black_share_withinteraction', leaveout = leaveout_except(columns, keep = keep))
table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], sweep_interactions = LOGIT_INTERACTIONS, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:slides/predicted_outcomes/black_share_interaction_sweep', column_labels = sweep_cols)
table = predicted_outcomes_by_stratum_from_fit(model, ind_sample, x_vars, columns, link = 'logit', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], bins = 4, sweep_label = 'Est. Highway Suitability', sweep_var = 'logit_normalized', sweep_interactions = LOGIT_INTERACTIONS)

# notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 1 of Table \\ref{tab:results/black_share}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values, including the indicator for the presence of any Black residents, " \
# "while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
# table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], sweep_interactions = [], sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
# export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/black_share_sweep', notes = notes)

# keep = [label for _, label in CORE + CNN_LOGIT + LOGIT_INTERACTIONS]
# table1 = format_regression_results(model)
# print(table1)
# table1 = table1.rename(index={
#     'Black': 'Exposure to Black Population',
#     'Residential x Black': 'Residential x Exposure to Black Population',
#     'Logit': 'Est. Highway Suitability',
#     'Residential x Logit': 'Residential x Est. Highway Suitability',
#     'Black x Logit': 'Exposure to Black Population x Est. Highway Suitability',
#     'Residential x Black x Logit': 'Residential x Exposure to Black Population x Est. Highway Suitability',
# })
# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
# model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var='hwy', cutoff_m=1500)
# notes = "Predicted outcomes from the Firth's bias-reduced logistic regression in Column 2 of Table \\ref{tab:results/black_share}, evaluated separately for Residential/Non-Residential areas and neighborhoods at the 25th/75th percentile of Exposure to Black Residents. All covariates other than Est. Highway Suitability are held constant at their own values, including the indicator for the presence of any Black residents, " \
# "while predictions are reported at the 25th, 50th, 75th, and 90th percentile of the distribution of Highway Suitability values for squares containing a highway. Standard errors are computed via the delta method from the spatial covariance matrix. The Protection Effect is the difference in predicted rates between Residential and Non-Residential areas at each level of exposure to Black Residents. *** p < 0.01, ** p < 0.05, * p < 0.1"
# table = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', black_values = [ind_sample['dem_access'].quantile(0.25), ind_sample['dem_access'].quantile(0.75)], black_labels = ['Low Exposure', 'High Exposure'], sweep_interactions = LOGIT_INTERACTIONS, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_values = sweep_values)
# export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:predicted_outcomes/black_share_sweep_logitinteraction', notes = notes)