import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.latex_formatting import export_single_regression
from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates, compute_demographic_access
)
from analysis.lib.bootstrap import bootstrap_ppml_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, PROB_INTERACTIONS, LOGIT_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS, DEM_ACCESS, PCT_BLACK, SHARE_BLACK, CNN_LOGIT,
    build_spec, leaveout_except,
)
from analysis.lib.marginal_effects import marginal_effects_table
from data_code.candidates import candidate_dict

df = load_sample()
df = restrict_to_discretionary(df)
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)

df = compute_demographic_access(df, 'pct_black', decay_m = 300, max_dist_m = 5000)
x_vars, columns = build_spec(df, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS)
x_vars_dem, columns_dem = build_spec(df, DEM_ACCESS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT)

# bootstrap_ppml_table mirrors bootstrap_lpm_table's (table, beta, se, boot_coefs) shape --
# see analysis/lib/bootstrap.py module docstring -- so this call is a drop-in swap for
# bootstrap_lpm_table/fit_ols if you want to try a different fit method here.
results, beta, se, boot_coefs = bootstrap_ppml_table(df, x_vars, columns, n_bootstraps=500, seed=42)
notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement, estimated using a Poisson Pseudo-Log-Linear model. The sample is restricted to a subset of grid squares designated as discretionary. " \
"The discretionary sample excludes grid squares that are intersected by a highway in 1940 or are directly adjacent to a highway in 1940. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 500 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
"This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model. * p<0.10, ** p<0.05, *** p<0.01"
print(results)
export_single_regression(
    results,
    caption='Determinants of Highway Placement - PPML on Discretionary Sample',
    label='tab:ppml_results',
    widthmultiplier=0.7,
    leaveout=leaveout_except(columns, keep=[label for _, label in CORE_VARS]),
    notes=notes
)



# dir_sample, ind_sample = split_by_candidates(df, candidate_dict)
# dir_results, dir_beta, dir_se, dir_boot_coefs = bootstrap_ppml_table(dir_sample, x_vars, columns, n_bootstraps=500, seed=42)
# ind_results, ind_beta, ind_se, ind_boot_coefs = bootstrap_ppml_table(ind_sample, x_vars, columns, n_bootstraps=500, seed=42)


# marginal_effects_table works the same way here as it does for OLS/LPM (see
# cnn_specif.py) -- just pass link='log' (PPML has an exponential mean function) plus the
# sweep_* args to sweep the CNN logit across its quantiles, interacted with Residential/Black:
# logit_percentiles = df['logit_hwy'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
# cells = marginal_effects_table(
#     df, x_vars, columns, beta, boot_coefs,
#     link='log',
#     sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values=logit_percentiles,
#     sweep_interactions=LOGIT_INTERACTIONS,
# )
