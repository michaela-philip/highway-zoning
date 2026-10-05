import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_characteristic_access, compute_shares
)
from analysis.lib.specs import (
    Spec, interact, cross, labels, leaveout_except, percentile_levels, city_dummies,
    RES, DEM_ACCESS, SUIT, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, HWY_ACCESS,
)
from analysis.lib.marginal_effects import Comparison, Sweep, Strata, predicted_outcomes, export_predicted_outcomes_table
from analysis.lib.estimators import fit_logit_firth_conley
from data_code.candidates import get_candidate_dict
from helpers.latex_formatting import export_single_regression, format_regression_results
from analysis.lib.figures import plot_access_map


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
CONTROLS = [HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS, HWY_ACCESS]
PROTECTION = Comparison(effect=RES, across=DEM_ACCESS, effect_name='Protection Effect', difference_name='Disparate Protection')
keep = labels(CORE, CNN_LOGIT, LOGIT_INTERACTIONS)

spec = Spec(CORE, CNN_LOGIT, CONTROLS)
model = fit_logit_firth_conley(ind_sample, spec, y_var='hwy', cutoff_m=1500)
table = format_regression_results(model)
export_single_regression(table, caption = 'Exposure to Black Population', label = 'tab:slides/results/black_share_nointeraction', leaveout = leaveout_except(spec.columns, keep = keep))
table = predicted_outcomes(model, ind_sample, PROTECTION, columns = SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:slides/predicted_outcomes/black_share_sweep')
table = predicted_outcomes(model, ind_sample, PROTECTION)

spec = Spec(CORE, CNN_LOGIT, LOGIT_INTERACTIONS, CONTROLS, city_dummies(ind_sample))
model = fit_logit_firth_conley(ind_sample, spec, y_var='hwy', cutoff_m=1500)
table = format_regression_results(model)
export_single_regression(table, caption = 'Exposure to Black Population', label = 'tab:slides/results/black_share_withinteraction', leaveout = leaveout_except(spec.columns, keep = keep))
table = predicted_outcomes(model, ind_sample, PROTECTION, columns = SUIT_SWEEP)
export_predicted_outcomes_table(table, caption = 'Predicted Highway Construction ', label = 'tab:slides/predicted_outcomes/black_share_interaction_sweep')
table = predicted_outcomes(model, ind_sample, PROTECTION, columns = Strata(SUIT, 4))
