import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates,
)
from analysis.lib.bootstrap import bootstrap_lpm_table, fit_ols
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, PROB_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS,
    build_spec,
)
from analysis.lib.marginal_effects import marginal_effects_table
from data_code.candidates import candidate_dict

df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)

x_vars_no_int, columns_no_int = build_spec(df, CORE_VARS, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
x_vars, columns = build_spec(df, CORE_VARS, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)

# direct vs. indirect samples (ML/manual candidate squares vs. the rest), interaction spec
dir_sample, ind_sample = split_by_candidates(df, candidate_dict)

dir_results_interaction, dir_beta_i, dir_se_i, dir_boot_coefs_i = bootstrap_lpm_table(dir_sample, x_vars, columns)
indir_results_interaction, ind_beta_i, ind_se_i, ind_boot_coefs_i = bootstrap_lpm_table(ind_sample, x_vars, columns)
print('direct sample with interactions:' , dir_results_interaction)
print('indirect sample with interactions:' , indir_results_interaction)

dir_results, dir_beta, dir_se, dir_boot_coefs = bootstrap_lpm_table(dir_sample, x_vars_no_int, columns_no_int)
indir_results, ind_beta, ind_se, ind_boot_coefs = bootstrap_lpm_table(ind_sample, x_vars_no_int, columns_no_int)

print('direct sample without interactions:' , dir_results)
print('indirect sample without interactions:' , indir_results)

# --- run for both specs ---
cells1 = marginal_effects_table(dir_sample, x_vars, columns, dir_beta_i, dir_boot_coefs_i)
cells2 = marginal_effects_table(ind_sample, x_vars, columns, ind_beta_i, ind_boot_coefs_i)