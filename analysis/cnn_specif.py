import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.latex_formatting import format_regression_results
from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates,
)
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, CNN_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS,
    build_spec, fit_ols,
)
from analysis.lib.marginal_effects import marginal_effects_table
from data_code.candidates import candidate_dict

df = load_sample()
df_restricted = restrict_to_discretionary(df)
df_restricted = merge_cnn_probs(df_restricted, 'predicted_activation-model1*.csv', dataroot='cnn/')
df_restricted = add_cnn_interactions(df_restricted)

# whole sample - no interaction with prob_hwy
x_vars_no_int, columns_no_int = build_spec(df_restricted, CORE_VARS, CNN_PROB, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
results_wholesample = format_regression_results(fit_ols(df_restricted, x_vars_no_int, columns_no_int))
print('wholesample without interactions:', results_wholesample)

# whole sample - interaction with prob_hwy
x_vars, columns = build_spec(df_restricted, CORE_VARS, CNN_PROB, CNN_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
results_wholesample_interaction = format_regression_results(fit_ols(df_restricted, x_vars, columns))
print('wholesample with interactions:', results_wholesample_interaction)

# direct vs. indirect samples (ML/manual candidate squares vs. the rest), interaction spec
dir_sample, ind_sample = split_by_candidates(df_restricted, candidate_dict)
dir_results_interaction, dir_beta, dir_se, dir_boot_coefs = bootstrap_lpm_table(dir_sample, x_vars, columns)
indir_results_interaction, ind_beta_i, ind_se_i, ind_boot_coefs_i = bootstrap_lpm_table(ind_sample, x_vars, columns)

indir_results_no_interaction, ind_beta, ind_se, ind_boot_coefs = bootstrap_lpm_table(ind_sample, x_vars_no_int, columns_no_int)

# --- run for both specs ---
print("\n=== WITHOUT CNN LOGIT ===")
cells1 = marginal_effects_table(ind_sample, x_vars_no_int, columns_no_int, ind_beta, ind_boot_coefs)

print("\n=== WITH CNN LOGIT ===")
cells2 = marginal_effects_table(ind_sample, x_vars, columns, ind_beta_i, ind_boot_coefs_i)
