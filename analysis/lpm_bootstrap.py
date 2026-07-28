import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, add_cnn_interactions
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.marginal_effects import marginal_effects_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS,
    build_spec, leaveout_except,
)
from data_code.candidates import candidate_dict

df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)
df_restricted = restrict_to_discretionary(df)

# whole sample - no interaction with hwy_logit
x_vars_no_int, columns_no_int = build_spec(df, CORE_VARS, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
whole_results, *_ = bootstrap_lpm_table(df, x_vars_no_int, columns_no_int)
print('whole sample without interactions:', whole_results)

# whole sample - interaction with hwy logit
x_vars, columns = build_spec(df, CORE_VARS, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
whole_int_results, *_ = bootstrap_lpm_table(df, x_vars, columns)
print('whole sample with interactions:', whole_int_results)

# discretionary sample - no interaction with hwy_logit
discretionary_results, *_ = bootstrap_lpm_table(df_restricted, x_vars_no_int, columns_no_int)
print('discretionary sample without interactions:', discretionary_results)

# discretionary sample - interaction with hwy_logit
discretionary_int_results, beta, se, boot_coefs = bootstrap_lpm_table(df_restricted, x_vars, columns)
print('discretionary sample with interactions:', discretionary_int_results)

# marginal effects 
sweep_values = df_restricted['logit_hwy'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
cells = marginal_effects_table(df_restricted, x_vars, columns, beta, boot_coefs, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_interactions=LOGIT_INTERACTIONS, sweep_values = sweep_values)

# # ML-predicted sample: restrict to grid squares with a well-identified CNN probability
# df_ml = merge_cnn_probs(df_restricted, 'predicted_activation-model4*.csv', dataroot='cnn/')
# cutoff_low = df_ml.loc[df_ml['prob_hwy'].notnull(), 'prob_hwy'].quantile(0.05)
# cutoff_high = df_ml.loc[df_ml['prob_hwy'].notnull(), 'prob_hwy'].quantile(0.95)
# df_ml['dm_prob'] = df_ml.groupby('city')['prob_hwy'].transform(lambda x: (x - x.mean()) / x.std())
# ml_sample = df_ml.loc[(df_ml['prob_hwy'] >= cutoff_low) & (df_ml['prob_hwy'] <= cutoff_high)].copy()
# ml_results, *_ = bootstrap_lpm_table(ml_sample, x_vars, columns)

# export_multiple_regressions(
#     {"Direct Sample": dir_results, "Indirect Sample": indir_results, "ML Sample": ml_results},
#     caption="Determinants of Highway Placement - Manual Sample Restriction",
#     label='tab:sample_restrict',
#     leaveout=leaveout,
# )

# # stratify by predicted probability of highway placement (demeaned within city)
# hwy_cutoff = ml_sample.loc[ml_sample['hwy'] == 1, 'dm_prob'].mean()
# controls = ml_sample[(ml_sample['hwy'] == 0) & (ml_sample['hwy_40'] == 0)].copy()
# df_high = pd.concat([ml_sample[ml_sample['dm_prob'] >= hwy_cutoff], controls], ignore_index=True)
# df_low = pd.concat([ml_sample[ml_sample['dm_prob'] < hwy_cutoff], controls], ignore_index=True)

# high_results, *_ = bootstrap_lpm_table(df_high, x_vars, columns)
# low_results, *_ = bootstrap_lpm_table(df_low, x_vars, columns)
# export_multiple_regressions(
#     {"High Predicted Probability": high_results, "Low Predicted Probability": low_results},
#     caption="Determinants of Highway Placement - Stratified by Predicted Probability",
#     label='tab:sample_stratified_probability',
#     leaveout=leaveout,
# )

# # stratify by distance to existing (1940) highway infrastructure
# dist_cutoff = df_ml['dist_to_hwy'].median()
# df_close = df_ml[df_ml['dist_to_hwy'] <= dist_cutoff]
# df_far = df_ml[df_ml['dist_to_hwy'] > dist_cutoff]
# close_results, *_ = bootstrap_lpm_table(df_close, x_vars, columns)
# far_results, *_ = bootstrap_lpm_table(df_far, x_vars, columns)
# export_multiple_regressions(
#     {"Close to Existing Hwy": close_results, "Far from Existing Hwy": far_results},
#     caption="Determinants of Highway Placement - Stratified by Distance to Existing Highway",
#     label='tab:sample_stratified_distance',
#     leaveout=leaveout,
# )
