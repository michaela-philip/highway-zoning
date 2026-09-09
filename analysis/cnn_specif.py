import sys
from pathlib import Path
import statsmodels.api as sm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, compute_demographic_access
)
from analysis.lib.bootstrap import bootstrap_lpm_table, fit_ols
from analysis.lib.specs import (
    HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, CNN_LOGIT,
    build_spec, leaveout_except, core_spec, sweep_interactions_spec, black_definition_note,
)
from analysis.lib.marginal_effects import export_predicted_outcomes_table, predicted_outcomes_from_fit
from data_code.candidates import candidate_dict
from helpers.latex_formatting import export_single_regression, export_multiple_regressions

cell_width = 200
df = load_sample(cell_width, impute = True)
df = merge_cnn_probs(df, f'predicted_activation-model1_{cell_width}*.csv', dataroot='cnn/')
df = restrict_to_discretionary(df)
df = compute_demographic_access(df, 'pct_black', decay_m = 400, max_dist_m = 3000)
df = df[df['imputed'] == 0]

dir_sample, ind_sample = split_by_candidates(df, candidate_dict)
sweep_values = ind_sample['logit_hwy'].quantile([0.25, 0.50, 0.75]).tolist()


CORE = core_spec(ind_sample, '60pct')
LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, '60pct', 'logit_hwy', 'CNN Logit')
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
model = sm.GLM(ind_sample['hwy'], sm.add_constant(ind_sample[x_vars]), family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
print(model.summary())
predicted_outcomes = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'log')

CORE = core_spec(ind_sample, 'dem_access')
LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, 'dem_access', 'logit_hwy', 'CNN Logit')
x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
model = sm.GLM(ind_sample['hwy'], sm.add_constant(ind_sample[x_vars]), family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
print(model.summary())
predicted_outcomes = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, link = 'log')

# CORE = core_spec(ind_sample, '50pct')
# LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, '50pct', 'logit_hwy', 'CNN Logit')
# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], sm.add_constant(ind_sample[x_vars]), family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# predicted_outcomes = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values=sweep_values, sweep_interactions=LOGIT_INTERACTIONS, link = 'log')


# CORE = core_spec(ind_sample, '40pct')
# LOGIT_INTERACTIONS = sweep_interactions_spec(ind_sample, '40pct', 'logit_hwy', 'CNN Logit')
# x_vars, columns = build_spec(ind_sample, CORE, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], sm.add_constant(ind_sample[x_vars]), family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# predicted_outcomes = predicted_outcomes_from_fit(model, ind_sample, x_vars, columns, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values=sweep_values, sweep_interactions=LOGIT_INTERACTIONS, link = 'log')



# df = compute_demographic_access(df, 'pct_black', 500, 0.019, 3000)

# x_vars, columns = build_spec(df, DEM_ACCESS, DEM_ACCESS_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, IMPUTED)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())

# x_vars, columns = build_spec(df, CORE_VARS, LOGIT_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS, IMPUTED)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())

# x_vars, columns = build_spec(df, DEM_ACCESS, DEM_ACCESS_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# res = fit_ppml_conley(ind_sample, x_vars, columns, cutoff_m=3000)
# print(format_regression_results(res))

# x_vars, columns = build_spec(df, CORE_VARS, LOGIT_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# res = fit_ppml_conley(ind_sample, x_vars, columns, cutoff_m=3000)
# print(format_regression_results(res))

# x_vars, columns = build_spec(df, )

# ind_sample = ind_sample[ind_sample['imputed'] == 0]

# x_vars, columns = build_spec(df, DEM_ACCESS, DEM_ACCESS_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# res = fit_ppml_conley(ind_sample, x_vars, columns, cutoff_m=3000)
# print(format_regression_results(res))

# x_vars, columns = build_spec(df, CORE_VARS, LOGIT_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())
# res = fit_ppml_conley(ind_sample, x_vars, columns, cutoff_m=3000)
# print(format_regression_results(res))

# x_vars, columns = build_spec(df, DEM_ACCESS, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())

# x_vars, columns = build_spec(df, CORE_VARS, LOGIT_INTERACTIONS, CNN_LOGIT, HOUSING_VARS, HH_CONTROLS, LOG_DIST_HWY, GEO_CONTROLS)
# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())

# # direct vs. indirect samples (ML/manual candidate squares vs. the rest), interaction spec
# dir_sample, ind_sample = split_by_candidates(df, candidate_dict)

# model = sm.GLM(ind_sample['hwy'], ind_sample[x_vars], family=sm.families.Poisson(link=sm.families.links.Log())).fit(cov_type='HC3', maxiter=200)
# print(model.summary())

# res = fit_ppml_conley(ind_sample, x_vars, columns, cutoff_m=3000)
# print(format_regression_results(res))

# dir_results_interaction, dir_beta_i, dir_se_i, dir_boot_coefs_i = bootstrap_lpm_table(dir_sample, x_vars, columns)
# indir_results_interaction, ind_beta_i, ind_se_i, ind_boot_coefs_i = bootstrap_lpm_table(ind_sample, x_vars, columns)

# # dir_results, dir_beta, dir_se, dir_boot_coefs = bootstrap_lpm_table(dir_sample, x_vars_no_int, columns_no_int)
# indir_results, ind_beta, ind_se, ind_boot_coefs = bootstrap_lpm_table(ind_sample, x_vars_no_int, columns_no_int)

# # --- run for both specs ---
# sweep_values = [ind_sample['logit_hwy'].mean()]
# cells = marginal_effects_table(ind_sample, x_vars, columns, ind_beta_i, ind_boot_coefs_i, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values = sweep_values, sweep_interactions=LOGIT_INTERACTIONS)

# keep = [label for _, label in CORE_VARS + CNN_LOGIT + LOGIT_INTERACTIONS]

# notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are outside of " \
# "the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
# "This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model. * p<0.10, ** p<0.05, *** p<0.01"
# export_single_regression(indir_results, caption = 'Determinants of Highway Placement - Outside Highway Corridor with Uninteracted CNN Logit', label = 'tab:indirect_results_no_interaction', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

# notes = "This table contains estimates of the imapct of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are inside of " \
# "the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
# "This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model, as well as the interaction between this estimated logit and the variables of interest. * p<0.10, ** p<0.05, *** p<0.01" 
# export_single_regression(dir_results_interaction, caption = 'Determinants of Highway Placement - Inside Highway Corridor with Interacted CNN Logit', label = 'tab:direct_results', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

# notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are outside of " \
# "the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
# "This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model, as well as the interaction between this estimated logit and the variables of interest. * p<0.10, ** p<0.05, *** p<0.01" 
# export_single_regression(indir_results_interaction, caption= 'Determinants of Highway Placement - Outside Highway Corridor with Interacted CNN Logit', label = 'tab:indirect_results', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

# notes = "This table contains predicted outcomes for each square in the sample based on their residential and majority-Black status, holding all other variables at their mean. The marginal effect of residential zoning and majority-Black status " \
# "is calculated as the difference in predicted outcomes between squares with and without these characteristics. These marginal effects are calculated based on the coefficients reported in Table \\ref{tab:indirect_results}. * p<0.10, ** p<0.05, *** p<0.01"
# col_name = ['Mean Logit Value']
# column_labels = dict(zip(sweep_values, col_name))
# export_marginal_effects_table(cells, caption = 'Marginal Effects of Residential Zoning and Majority-Black Status on Highway Placement - Outside Highway Corridor', label = 'tab:marginal_effects_indir', column_labels = column_labels, widthmultiplier=0.6, notes = notes)

# sweep_values = ind_sample['logit_hwy'].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).tolist()
# col_name = ['10th Percentile', '25th Percentile', '50th Percentile', '75th Percentile', '90th Percentile']
# column_labels = dict(zip(sweep_values, col_name))
# cells = marginal_effects_table(ind_sample, x_vars, columns, ind_beta_i, ind_boot_coefs_i, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values = sweep_values, sweep_interactions=LOGIT_INTERACTIONS)
# notes = "This table contains predicted outcomes for each square in the sample based on their residential and majority-Black status, holding all variables except for the CNN Logit at their mean. Each column contains " \
# "the results of a different prediction where the CNN logit is varied across its quantiles. The marginal effect of residential zoning and majority-Black status " \
# "is calculated as the difference in predicted outcomes between squares with and without these characteristics. These marginal effects are calculated based on the coefficients reported in Table \\ref{tab:indirect_results}. * p<0.10, ** p<0.05, *** p<0.01"
# export_marginal_effects_table(cells, caption="Marginal Effects of Residential Zoning and Majority-Black Status - CNN Quantiles", label='tab:marginal_effects_sweep', column_labels = column_labels, widthmultiplier = 1, notes=notes)