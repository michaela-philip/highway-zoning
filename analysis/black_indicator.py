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
print(format_regression_results(model))
predicted_outcomes_by_stratum_from_fit(model, ind_sample, x_vars, columns, link = 'logit', sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS, bins = sweep_bins)
predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = LOGIT_INTERACTIONS)


x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_RACE_CONTROLS, GEO_CONTROLS, HWY_ACCESS, include_city_dummies=False)
model = fit_logit_firth_conley(ind_sample, x_vars, columns, y_var = 'hwy', cutoff_m = 1500)
print(format_regression_results(model))
predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'logit', eval_at = 'ame', sweep_values = sweep_values, sweep_var = 'logit_normalized', sweep_label = 'Est. Highway Suitability', sweep_interactions = [])

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