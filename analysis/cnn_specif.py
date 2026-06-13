import sys
import glob
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import geopandas as gpd

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, bootstrap_results_to_namespace
from data_code.candidates import candidate_dict

df = pd.read_pickle('data/output/sample.pkl')

# identify squares adjacent to existing highways
hwy_40_squares = df[df['hwy_40'] == 1][['grid_id', 'geometry']].copy()
all_squares = df[['grid_id', 'geometry']].copy()
touches_result = gpd.sjoin(
    all_squares,
    hwy_40_squares[['geometry']],
    how='left',
    predicate='touches'
)
# squares that got a match are adjacent to the 1940 network
adjacent_ids = set(
    touches_result[touches_result['index_right'].notna()]['grid_id']
)
df_restricted = df[~df['grid_id'].isin(adjacent_ids)].copy()
df_restricted = df[df['hwy_40'] == 0].copy() # drop any remaining 1940 highways that are not adjacent to others

# merge in cnn probabilities
dataroot = 'cnn/'
model1 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model1*.csv')), key=os.path.getmtime, reverse=True)[0]
logits_df = pd.read_csv(model1)
logits_df['grid_id'] = logits_df['grid_id'].astype(str)
df_restricted['grid_id'] = df_restricted['grid_id'].astype(str)
df_restricted = df_restricted.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
df_restricted['grid_id'] = df_restricted['grid_id'].astype(int)

# prepare variables
df_restricted['rent'] = df_restricted['rent'].replace(0, 0.00001)
df_restricted['valueh'] = df_restricted['valueh'].replace(0, 0.00001)
df_restricted['log_valueh'] = np.log(df_restricted['valueh']) * df_restricted['valueh_avail']
df_restricted['log_rent'] = np.log(df_restricted['rent']) * df_restricted['rent_avail']
df_restricted['city_louisville'] = (df_restricted['city'] == 'louisville').astype(int)
df_restricted['city_littlerock'] = (df_restricted['city'] == 'littlerock').astype(int)
df_restricted['distance_to_cbd_sq'] = df_restricted['distance_to_cbd'] ** 2
df_restricted['log_dist_to_rr'] = np.log(df_restricted['dist_to_rr'])
df_restricted['log_dist_to_rr_sq'] = df_restricted['log_dist_to_rr'] ** 2
df_restricted['log_dist_to_hwy'] = np.log(df_restricted['dist_to_hwy'])
df_restricted['ResidentialxBlack'] = df_restricted['Residential'] * df_restricted['mblack_1945def']
df_restricted['ResidentialxBlackxProbHwy'] = df_restricted['Residential'] * df_restricted['mblack_1945def'] * df_restricted['prob_hwy']
df_restricted['BlackxProbHwy'] = df_restricted['mblack_1945def'] * df_restricted['prob_hwy']
df_restricted['ResidentialxProbHwy'] = df_restricted['Residential'] * df_restricted['prob_hwy']


x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'BlackxProbHwy', 'ResidentialxProbHwy', 'ResidentialxBlackxProbHwy', 'prob_hwy', 'log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Black x Probability of Highway', 'Residential x Probability of Highway', 'Residential x Black x Probability of Highway', 'Probability of Highway (CNN)', 'Log(Value)', 'Log(Rent)', 'Log(Distance to Highway)', 'Owner', 'Number of Residents', 'City_Louisville', 'City_LittleRock']

# results_wholesample = format_regression_results(sm.OLS(df_restricted['hwy'], sm.add_constant(df_restricted[x_vars])).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['city']}))
# print(results_wholesample)
# export_single_regression(results_wholesample, caption = 'Determinants of Highway Placement - Full Sample', label = 'tab:cnn_covariate', widthmultiplier=0.7, leaveout = ['log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'])

def bootstrap_lpm(sample, x_vars, n_bootstraps=1000, seed = 42):
    rng = np.random.default_rng(seed)

    n = len(sample)
    y = sample['hwy'].values
    X = np.column_stack([np.ones(len(sample)), sample[x_vars].values])
    
    print(X.shape, y.shape) 

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    boot_coefs = []
    for b in range(n_bootstraps):
        boot_idx = rng.choice(n, size = n, replace = True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        beta_boot = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
        boot_coefs.append(beta_boot)    
    
    boot_coefs = np.array(boot_coefs)
    se = boot_coefs.std(axis = 0)
    ci_lower = np.percentile(boot_coefs, 2.5, axis = 0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis = 0)
    return beta_hat, boot_coefs, se, ci_lower, ci_upper, y, X

x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'prob_hwy', 'BlackxProbHwy', 'ResidentialxProbHwy', 'ResidentialxBlackxProbHwy', 'log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Probability of Highway (CNN)', 'Black x Probability of Highway', 'Residential x Probability of Highway', 'Residential x Black x Probability of Highway', 'Log(Value)', 'Log(Rent)', 'Log(Distance to Highway)', 'Owner', 'Number of Residents', 'City_Louisville', 'City_LittleRock']

# whole sample - no interaction with prob_hwy
x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'prob_hwy', 'log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Probability of Highway (CNN)', 'Log(Value)', 'Log(Rent)', 'Log(Distance to Highway)', 'Owner', 'Number of Residents', 'City_Louisville', 'City_LittleRock']
results_wholesample = format_regression_results(sm.OLS(df_restricted['hwy'], sm.add_constant(df_restricted[x_vars])).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['city']}))
print('wholesample without interactions:', results_wholesample)

# whole sample - interaction with prob_hwy
x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'prob_hwy', 'BlackxProbHwy', 'ResidentialxProbHwy', 'ResidentialxBlackxProbHwy', 'log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Probability of Highway (CNN)', 'Black x Probability of Highway', 'Residential x Probability of Highway', 'Residential x Black x Probability of Highway', 'Log(Value)', 'Log(Rent)', 'Log(Distance to Highway)', 'Owner', 'Number of Residents', 'City_Louisville', 'City_LittleRock']
results_wholesample_interaction = format_regression_results(sm.OLS(df_restricted['hwy'], sm.add_constant(df_restricted[x_vars])).fit(cov_type='cluster', cov_kwds={'groups': df_restricted['city']}))
print('wholesample with interactions:', results_wholesample_interaction)

# direct sample - interaction
out_frames = []
for city in df_restricted['city'].unique():
    candidates = candidate_dict[city]
    controls = df_restricted.loc[(df_restricted['city'] == city) & (df_restricted['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
dir_sample = pd.concat(out_frames, ignore_index=True)
# dir_sample['grid_id'] = dir_sample['grid_id'].astype(str)
# dir_sample = dir_sample.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
# print(dir_sample.shape)
dir_beta, dir_boot_coefs, dir_se, dir_ci_lower, dir_ci_upper, y, X = bootstrap_lpm(dir_sample, x_vars)
dir_results_interaction = bootstrap_results_to_namespace(dir_beta, dir_boot_coefs, y, X, col_names = columns)
dir_results_interaction = format_regression_results(dir_results_interaction)

# indirect sample - interaction
out_frames = []
for city in df_restricted['city'].unique():
    candidates = candidate_dict[city]
    controls = df_restricted.loc[(df_restricted['city'] == city) & (~df_restricted['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
ind_sample = pd.concat(out_frames, ignore_index=True)
# ind_sample['grid_id'] = ind_sample['grid_id'].astype(str)
# ind_sample = ind_sample.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
ind_beta, ind_boot_coefs, ind_se, ind_ci_lower, ind_ci_upper, y, X = bootstrap_lpm(ind_sample, x_vars)
indir_results_interaction = bootstrap_results_to_namespace(ind_beta, ind_boot_coefs, y, X, col_names = columns)
indir_results_interaction = format_regression_results(indir_results_interaction)

x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'prob_hwy', 'log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Probability of Highway (CNN)', 'Log(Value)', 'Log(Rent)', 'Log(Distance to Highway)', 'Owner', 'Number of Residents', 'City_Louisville', 'City_LittleRock']

# direct sample - no interaction
dir_beta,dir_boot_coefs, dir_se, dir_ci_lower, dir_ci_upper, y, X = bootstrap_lpm(dir_sample, x_vars)
dir_results_no_interaction = bootstrap_results_to_namespace(dir_beta, dir_boot_coefs, y, X, col_names = columns)
dir_results_no_interaction = format_regression_results(dir_results_no_interaction)

#indirect sample - no interaction
ind_beta, ind_boot_coefs, ind_se, ind_ci_lower, ind_ci_upper, y, X = bootstrap_lpm(ind_sample, x_vars)
indir_results_no_interaction = bootstrap_results_to_namespace(ind_beta, ind_boot_coefs, y, X, col_names = columns)
indir_results_no_interaction = format_regression_results(indir_results_no_interaction)

export_multiple_regressions({'Whole Sample': results_wholesample, 'Direct Sample': dir_results_no_interaction, 'Indirect Sample': indir_results_no_interaction}, caption = 'Determinants of Highway Placement - Conditional on CNN (No Interactions)', label = 'tab:cnn_covariate_no_interaction', leaveout = ['log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'])

export_multiple_regressions({'Whole Sample': results_wholesample_interaction, 'Direct Sample': dir_results_interaction, 'Indirect Sample': indir_results_interaction}, caption = 'Determinants of Highway Placement - Conditional on CNN (With Interactions)', label = 'tab:cnn_covariate_interaction', leaveout = ['log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'])
# export_multiple_regressions({'Whole Sample': results_wholesample, 'Whole Sample with Interactions': results_wholesample_interaction, 
#                              'Direct Sample with Interactions': dir_results_interaction, 'Indirect Sample with Interactions': indir_results_interaction, 
#                              'Direct Sample without Interactions': dir_results_no_interaction, 'Indirect Sample without Interactions': indir_results_no_interaction}, caption = 'Determinants of Highway Placement - Conditional on CNN', label = 'tab:cnn_covariate', widthmultiplier=0.7, leaveout = ['log_valueh', 'log_rent', 'log_dist_to_hwy', 'owner', 'numprec', 'city_louisville', 'city_littlerock'])