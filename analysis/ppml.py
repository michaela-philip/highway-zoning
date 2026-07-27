import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.latex_formatting import export_single_regression
from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates, compute_demographic_access
)
from analysis.lib.bootstrap import bootstrap_ppml, bootstrap_ppml_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, PROB_INTERACTIONS, LOGIT_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS, DEM_ACCESS, PCT_BLACK, SHARE_BLACK, CNN_LOGIT,
    build_spec, leaveout_except,
)
from analysis.lib.marginal_effects import (marginal_effects_table, ppml_marginal_effects)
from data_code.candidates import candidate_dict

df = load_sample()
df = restrict_to_discretionary(df)
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)

df = compute_demographic_access(df, 'pct_black', decay_m = 600, max_dist_m = 5000)
x_vars, columns = build_spec(df, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS)

# bootstrap_ppml_table mirrors bootstrap_lpm_table's (table, beta, se, boot_coefs) shape --
# see analysis/lib/bootstrap.py module docstring -- so this call is a drop-in swap for
# bootstrap_lpm_table/fit_ols if you want to try a different fit method here.
results, beta, se, boot_coefs = bootstrap_ppml_table(df, x_vars, columns, n_bootstraps=500, seed=42)
print(results)
export_single_regression(
    results,
    caption='Determinants of Highway Placement - PPML',
    label='tab:ppml_results',
    widthmultiplier=0.7,
    leaveout=leaveout_except(columns, keep=[label for _, label in CORE_VARS]),
)



# dir_sample, ind_sample = split_by_candidates(df, candidate_dict)
# dir_results, dir_beta, dir_se, dir_boot_coefs = bootstrap_ppml_table(dir_sample, x_vars, columns, n_bootstraps=500, seed=42)
# ind_results, ind_beta, ind_se, ind_boot_coefs = bootstrap_ppml_table(ind_sample, x_vars, columns, n_bootstraps=500, seed=42)


# marginal effects (delta method) need the analytic GLM fit object, which only the
# lower-level bootstrap_ppml returns (bootstrap_ppml_table only returns the table/beta/se/
# boot_coefs needed for export):
# beta, boot_coefs, se, ci_lower, ci_upper, y, X, full_model = bootstrap_ppml(df, x_vars, n_bootstraps=500, seed=42)
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
