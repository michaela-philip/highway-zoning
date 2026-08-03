import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import (
    load_sample, restrict_to_discretionary, merge_cnn_probs, add_cnn_interactions, split_by_candidates,
)
from analysis.lib.bootstrap import bootstrap_lpm_table, fit_ols
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, HH_CONTROLS, CNN_PROB, PROB_INTERACTIONS, LOG_DIST_HWY, GEO_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS,
    build_spec, leaveout_except
)
from analysis.lib.marginal_effects import marginal_effects_table, export_marginal_effects_table
from data_code.candidates import candidate_dict
from helpers.latex_formatting import export_single_regression, export_multiple_regressions

df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)

x_vars_no_int, columns_no_int = build_spec(df, CORE_VARS, CNN_LOGIT, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)
x_vars, columns = build_spec(df, CORE_VARS, CNN_LOGIT, LOGIT_INTERACTIONS, HOUSING_VARS, LOG_DIST_HWY, HH_CONTROLS, GEO_CONTROLS)

# direct vs. indirect samples (ML/manual candidate squares vs. the rest), interaction spec
dir_sample, ind_sample = split_by_candidates(df, candidate_dict)

dir_results_interaction, dir_beta_i, dir_se_i, dir_boot_coefs_i = bootstrap_lpm_table(dir_sample, x_vars, columns)
indir_results_interaction, ind_beta_i, ind_se_i, ind_boot_coefs_i = bootstrap_lpm_table(ind_sample, x_vars, columns)

# dir_results, dir_beta, dir_se, dir_boot_coefs = bootstrap_lpm_table(dir_sample, x_vars_no_int, columns_no_int)
indir_results, ind_beta, ind_se, ind_boot_coefs = bootstrap_lpm_table(ind_sample, x_vars_no_int, columns_no_int)

# --- run for both specs ---
cells = marginal_effects_table(ind_sample, x_vars, columns, ind_beta_i, ind_boot_coefs_i, sweep_var='logit_hwy', sweep_label='CNN Logit', sweep_values = [ind_sample['logit_hwy'].mean()], sweep_interactions=LOGIT_INTERACTIONS)

keep = [label for _, label in CORE_VARS + CNN_LOGIT + LOGIT_INTERACTIONS]

notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are outside of " \
"the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
"This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model. * p<0.10, ** p<0.05, *** p<0.01"
export_single_regression(indir_results, caption = 'Determinants of Highway Placement - Outside Highway Corridor with Uninteracted CNN Logit', label = 'tab:indirect_results_no_interaction', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

notes = "This table contains estimates of the imapct of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are inside of " \
"the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
"This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model, as well as the interaction between this estimated logit and the variables of interest. * p<0.10, ** p<0.05, *** p<0.01" 
export_single_regression(dir_results_interaction, caption = 'Determinants of Highway Placement - Inside Highway Corridor with Interacted CNN Logit', label = 'tab:direct_results', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement. The sample is restricted to grid squares that are outside of " \
"the existing highway corridor. Standard errors are reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects. " \
"This model also includes a measure of a square's geographic suitability for highway placement, as estimated by a CNN model, as well as the interaction between this estimated logit and the variables of interest. * p<0.10, ** p<0.05, *** p<0.01" 
export_single_regression(indir_results_interaction, caption= 'Determinants of Highway Placement - Outside Highway Corridor with Interacted CNN Logit', label = 'tab:indirect_results', leaveout = leaveout_except(columns, keep=keep), widthmultiplier=0.6, notes = notes)

notes = "This table contains predicted outcomes for each square in the sample based on their residential and majority-Black status, holding all other variables at their mean. The marginal effect of residential zoning and majority-Black status " \
"is calculated as the difference in predicted outcomes between squares with and without these characteristics. These marginal effects are calculated based on the coefficients reported in Table \\ref{tab:indirect_results}. * p<0.10, ** p<0.05, *** p<0.01"
export_marginal_effects_table(cells, caption = 'Marginal Effects of Residential Zoning and Majority-Black Status on Highway Placement - Outside Highway Corridor', label = 'tab:marginal_effects_indir', widthmultiplier=0.6, notes = notes)