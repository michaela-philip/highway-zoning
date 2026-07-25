import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statsmodels.api as sm

from helpers.latex_formatting import format_regression_results
from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates, compute_demographic_access
)
from analysis.lib.bootstrap import (bootstrap_lpm_table, bootstrap_ppml, print_ppml_bootstrap_results)
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, PROB_INTERACTIONS, LOGIT_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS, DEM_ACCESS, PCT_BLACK, SHARE_BLACK, CNN_LOGIT,
    build_spec, fit_ols,
)
from analysis.lib.marginal_effects import (marginal_effects_table, ppml_marginal_effects)
from data_code.candidates import candidate_dict

df = load_sample()
df = restrict_to_discretionary(df)
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)

df = compute_demographic_access(df, 'pct_black', decay_m = 600, max_dist_m = 5000)
x_vars, columns = build_spec(df, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS)

beta, boot_coefs, se, ci_lower, ci_upper, columns, full_model = bootstrap_ppml(df, 'hwy', x_vars, n_bootstraps=500, seed = 42)
print_ppml_bootstrap_results(beta, boot_coefs, se, ci_lower, ci_upper, columns)



# dir_sample, ind_sample = split_by_candidates(df, candidate_dict)
# beta_ind, boot_coefs_ind, se_ind, ci_lower_ind, ci_upper_ind, columns_ind, full_model_ind = bootstrap_ppml(ind_sample, 'hwy', x_vars, n_bootstraps=500, seed = 42)
# print_ppml_bootstrap_results(beta_ind, boot_coefs_ind, se_ind, ci_lower_ind, ci_upper_ind, columns_ind)

# beta_dir, boot_coefs_dir, se_dir, ci_lower_dir, ci_upper_dir, columns_dir, full_model_dir = bootstrap_ppml(dir_sample, 'hwy', x_vars, n_bootstraps=500, seed = 42)
# print_ppml_bootstrap_results(beta_dir, boot_coefs_dir, se_dir, ci_lower_dir, ci_upper_dir, columns_dir)


# logit_percentiles = df['logit_hwy'].quantile(
#     [0.10, 0.25, 0.50, 0.75, 0.90]
# ).tolist()

# results = ppml_marginal_effects(
#     fitted_model      = full_model,
#     df                = df,
#     x_vars            = x_vars,
#     columns = columns,
#     eval_at           = 'mean',
#     logit_var         = 'logit_hwy',
#     logit_label       = 'LogHwy',
#     logit_eval_values = logit_percentiles
# )