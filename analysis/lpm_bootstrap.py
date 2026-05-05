import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import os
import matplotlib.pyplot as plt
import geopandas as gpd

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, bootstrap_results_to_namespace
from data_code.candidates import candidate_dict

df = pd.read_pickle('data/output/sample.pkl')
df['rent'] = df['rent'].replace(0, 0.00001)
df['valueh'] = df['valueh'].replace(0, 0.00001)

# prepare variables
df['log_valueh'] = np.log(df['valueh']) * df['valueh_avail']
df['log_rent'] = np.log(df['rent']) * df['rent_avail']
df['city_louisville'] = (df['city'] == 'louisville').astype(int)
df['city_littlerock'] = (df['city'] == 'littlerock').astype(int)
df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
df['log_dist_to_rr'] = np.log(df['dist_to_rr'])
df['log_dist_to_rr_sq'] = df['log_dist_to_rr'] ** 2
df['log_dist_to_hwy'] = np.log(df['dist_to_hwy'])
df['ResidentialxBlack'] = df['Residential'] * df['mblack_1945def']

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

# discretionary_ids = set(
#     hwy_new_squares[~hwy_new_squares['grid_id'].isin(adjacent_ids)]['grid_id']
# )
# hwys_discretionary = list(discretionary_ids)

df_restricted = df[~df['grid_id'].isin(adjacent_ids)].copy()
df_restricted = df[df['hwy_40'] == 0].copy() # drop any remaining 1940 highways that are not adjacent to others

x_vars = ['Residential', 'mblack_1945def', 'ResidentialxBlack', 'log_valueh', 'log_rent', 'log_dist_to_rr', 'log_dist_to_rr_sq', 'log_dist_to_hwy', 'distance_to_cbd', 'distance_to_cbd_sq', 'flood_risk', 'dist_water', 'slope', 'dm_elevation', 'owner', 'numprec', 'city_louisville', 'city_littlerock'] 
columns = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'dist(RR)', 'dist(RR^2)', 'ldist(Hwy)', 'dist(CBD)', 'dist(CBD^2)', 'Flood Risk', 'dist(Water)', 'Slope', 'Elevation', 'Owner', 'Number of Residents'] + [f'City_{c}' for c in df['city'].unique()[1:]]

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

# direct sample
out_frames = []
for city in df_restricted['city'].unique():
    candidates = candidate_dict[city]
    controls = df_restricted.loc[(df_restricted['city'] == city) & (df_restricted['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
dir_sample = pd.concat(out_frames, ignore_index=True)
dir_beta, dir_boot_coefs, dir_se, dir_ci_lower, dir_ci_upper, y, X = bootstrap_lpm(dir_sample, x_vars)
dir_results = bootstrap_results_to_namespace(dir_beta, dir_boot_coefs, y, X, col_names = columns)
dir_results = format_regression_results(dir_results)

# indirect sample
out_frames = []
for city in df_restricted['city'].unique():
    candidates = candidate_dict[city]
    controls = df_restricted.loc[(df_restricted['city'] == city) & (~df_restricted['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
ind_sample = pd.concat(out_frames, ignore_index=True)
ind_beta, ind_boot_coefs, ind_se, ind_ci_lower, ind_ci_upper, y, X = bootstrap_lpm(ind_sample, x_vars)
indir_results = bootstrap_results_to_namespace(ind_beta, ind_boot_coefs, y, X, col_names = columns)
indir_results = format_regression_results(indir_results)

# ML based sample 
# load CNN probabilities
dataroot = 'cnn/'
model4 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model4*.csv')), key=os.path.getmtime, reverse=True)[0]
logits_df = pd.read_csv(model4)
logits_df['grid_id'] = logits_df['grid_id'].astype(str)
df_restricted['grid_id'] = df_restricted['grid_id'].astype(str)
df = df_restricted.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
cutoff_low = df.loc[df['prob_hwy'].notnull(), 'prob_hwy'].quantile(0.05) 
cutoff_high = df.loc[df['prob_hwy'].notnull(), 'prob_hwy'].quantile(0.95)
df['dm_prob'] = df.groupby('city')['prob_hwy'].transform(lambda x: (x - x.mean()) / x.std())
# sample = df.loc[df['dm_prob'] > 0].copy()
sample = df.loc[(df['prob_hwy'] >= cutoff_low) & (df['prob_hwy']<= cutoff_high)].copy()

ml_beta, ml_boot_coefs, ml_se, ml_ci_lower, ml_ci_upper, y, X = bootstrap_lpm(sample, x_vars)
ml_results = bootstrap_results_to_namespace(ml_beta, ml_boot_coefs, y, X, col_names = columns)
ml_results = format_regression_results(ml_results)

# export direct and indirect results together
sample_restrict_table = export_multiple_regressions({"Direct Sample": dir_results, "Indirect Sample": indir_results, "ML Sample": ml_results},
                            caption = "Determinants of Highway Placement - Manual Sample Restriction",
                            label = 'tab:sample_restrict',
                            leaveout = ['Log(Value)', 'Log(Rent)', 'dist(RR)', 'dist(RR^2)', 'ldist(Hwy)', 'dist(CBD)', 'dist(CBD^2)', 'Flood Risk', 'dist(Water)', 'Slope', 'Elevation', 'Owner', 'Number of Residents'] + [f'City_{c}' for c in df['city'].unique()[1:]])

# stratify based on high and low probability
hwy_cutoff = sample.loc[sample['hwy'] == 1, 'dm_prob'].mean()
hwy_high = sample[(sample['dm_prob'] >= hwy_cutoff)]
hwy_low = sample[(sample['dm_prob'] < hwy_cutoff)]
controls = sample[(sample['hwy'] == 0) & (sample['hwy_40'] == 0)].copy()
df_high = pd.concat([hwy_high, controls], ignore_index=True)
df_low = pd.concat([hwy_low, controls], ignore_index=True)

high_beta, high_boot_coefs, high_se, high_ci_lower, high_ci_upper, y, X = bootstrap_lpm(df_high, x_vars)
high_results = bootstrap_results_to_namespace(high_beta, high_boot_coefs, y, X, col_names = columns)
high_results = format_regression_results(high_results)
low_beta, low_boot_coefs, low_se, low_ci_lower, low_ci_upper, y, X = bootstrap_lpm(df_low, x_vars)
low_results = bootstrap_results_to_namespace(low_beta, low_boot_coefs, y, X, col_names = columns)
low_results = format_regression_results(low_results)
stratified_table = export_multiple_regressions({"High Predicted Probability": high_results, "Low Predicted Probability": low_results},
                            caption = "Determinants of Highway Placement - Stratified by Predicted Probability",
                            label = 'tab:sample_stratified_probability',
                            leaveout = ['Log(Value)', 'Log(Rent)', 'dist(RR)', 'dist(RR^2)', 'ldist(Hwy)', 'dist(CBD)', 'dist(CBD^2)', 'Flood Risk', 'dist(Water)', 'Slope', 'Elevation', 'Owner', 'Number of Residents'] + [f'City_{c}' for c in df['city'].unique()[1:]])

# stratify based on distance to existing hwy infrastructure
cutoff = df['dist_to_hwy'].median()
df_close = df[df['dist_to_hwy'] <= cutoff]
df_far = df[df['dist_to_hwy'] > cutoff]
# df_close = pd.concat([df_close, controls], ignore_index=True)
# df_far = pd.concat([df_far, controls], ignore_index=True)
close_beta, close_boot_coefs, close_se, close_ci_lower, close_ci_upper, y, X = bootstrap_lpm(df_close, x_vars)
close_results = bootstrap_results_to_namespace(close_beta, close_boot_coefs, y, X, col_names = columns)
close_results = format_regression_results(close_results)
far_beta, far_boot_coefs, far_se, far_ci_lower, far_ci_upper, y, X = bootstrap_lpm(df_far, x_vars)
far_results = bootstrap_results_to_namespace(far_beta, far_boot_coefs, y, X, col_names = columns)
far_results = format_regression_results(far_results)
stratified_table = export_multiple_regressions({"Close to Existing Hwy": close_results, "Far from Existing Hwy": far_results},
                            caption = "Determinants of Highway Placement - Stratified by Distance to Existing Highway",
                            label = 'tab:sample_stratified_distance',
                            leaveout = ['Log(Value)', 'Log(Rent)', 'dist(RR)', 'dist(RR^2)', 'ldist(Hwy)', 'dist(CBD)', 'dist(CBD^2)', 'Flood Risk', 'dist(Water)', 'Slope', 'Elevation', 'Owner', 'Number of Residents'] + [f'City_{c}' for c in df['city'].unique()[1:]]) 

# run lpm with predicted probabilities as covariate on full sample
# sample = df.copy()
# eps = 1e-6
# prob_clipped = sample['prob_hwy'].clip(eps, 1 - eps)
# sample['logit'] = np.log(prob_clipped / (1 - prob_clipped))
# city_dummies = pd.get_dummies(sample['city'], prefix='city', drop_first=True).values
# X = np.column_stack([
#         np.ones(len(sample)),
#         sample['Residential'].values,
#         sample['mblack_1945def'].values,
#         sample['Residential'].values * sample['mblack_1945def'].values,
#         np.log(sample['valueh'].values),
#         np.log(sample['rent'].values),
#         sample['logit'].values,
#         sample['logit'].values**2,
#         sample['logit'].values**3,
#         city_dummies
#     ])
# y = sample['hwy'].values

# rng = np.random.default_rng(42)
# n = len(sample)
# n_bootstraps = 1000

# beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

# boot_coefs = []
# for b in range(n_bootstraps):
#     boot_idx = rng.choice(n, size = n, replace = True)
#     X_boot = X[boot_idx]
#     y_boot = y[boot_idx]
#     beta_boot = np.linalg.lstsq(X_boot, y_boot, rcond=None)[0]
#     boot_coefs.append(beta_boot)    

# boot_coefs = np.array(boot_coefs)
# se = boot_coefs.std(axis = 0)
# ci_lower = np.percentile(boot_coefs, 2.5, axis = 0)
# ci_upper = np.percentile(boot_coefs, 97.5, axis = 0)
# ml_results = bootstrap_results_to_namespace(beta_hat, boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Prob(Hwy)', 'Prob(Hwy)^2', 'Prob(Hwy)^3'] + [f'City_{c}' for c in sample['city'].unique()[1:]])
# ml_results = format_regression_results(ml_results)
# ml_cov_table = export_single_regression(ml_results, caption = "Determinants of Highway Placement - Full Sample with Predicted Probability", label = 'tab:ml_prob_cov', widthmultiplier=0.7, leaveout = ['owner', 'Intercept' + ''.join([f'City_{c}' for c in ind_sample['city'].unique()[1:]])])