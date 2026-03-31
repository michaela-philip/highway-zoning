import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import os

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, bootstrap_results_to_namespace
from data_code.candidates import candidate_dict

df = pd.read_pickle('data/output/sample.pkl')
df['rent'] = df['rent'].replace(0, np.nan)
df['valueh'] = df['valueh'].replace(0, np.nan)
df = df.dropna(subset = ['rent', 'valueh']).copy()

def bootstrap_lpm(sample, n_bootstraps=1000, seed = 42):
    rng = np.random.default_rng(seed)

    n = len(sample)

    # construct/transform some variables
    log_valueh = np.log(sample['valueh'].values)
    log_rent = np.log(sample['rent'].values)
    city_dummies = pd.get_dummies(sample['city'], prefix='city', drop_first=True).values
    owner = sample['owner'].values

    residential = sample['Residential'].values
    black = sample['mblack_1945def'].values
    X = np.column_stack([
        np.ones(n),
        residential,
        black,
        residential * black,
        log_valueh,
        log_rent,
        owner,
        city_dummies])
    y = sample['hwy'].values

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
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (df['grid_id'].isin(candidates))].copy()
    treated = df.loc[(df['city'] == city) & (df['hwy']==1) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
    out_frames.append(treated)
dir_sample = pd.concat(out_frames, ignore_index=True)
dir_beta, dir_boot_coefs, dir_se, dir_ci_lower, dir_ci_upper, y, X = bootstrap_lpm(dir_sample)
dir_results = bootstrap_results_to_namespace(dir_beta, dir_boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in dir_sample['city'].unique()[1:]])
dir_results = format_regression_results(dir_results)

# indirect sample
out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
ind_sample = pd.concat(out_frames, ignore_index=True)
ind_beta, ind_boot_coefs, ind_se, ind_ci_lower, ind_ci_upper, y, X = bootstrap_lpm(ind_sample)
indir_results = bootstrap_results_to_namespace(ind_beta, ind_boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in ind_sample['city'].unique()[1:]])
indir_results = format_regression_results(indir_results)

# ML based sample 
# load CNN probabilities
dataroot = 'cnn/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv_files.sort(key=os.path.getmtime, reverse=True)
logits_df = pd.read_csv(csv_files[0])
logits_df['grid_id'] = logits_df['grid_id'].astype(str)
df['grid_id'] = df['grid_id'].astype(str)
df = df.merge(logits_df[['grid_id', 'prob_hwy']], on='grid_id', how='left')
target_prob = df.loc[df['hwy'] == 1, 'prob_hwy'].quantile(0.25)
print(f"Target probability for ML-based sample restriction: {target_prob:.4f}")

sample = df.loc[df['prob_hwy'] >= target_prob].copy()
ml_beta, ml_boot_coefs, ml_se, ml_ci_lower, ml_ci_upper, y, X = bootstrap_lpm(sample)
ml_results = bootstrap_results_to_namespace(ml_beta, ml_boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in sample['city'].unique()[1:]])
ml_results = format_regression_results(ml_results)

# export direct and indirect results together
sample_restrict_table = export_multiple_regressions({"Direct Sample": dir_results, "Indirect Sample": indir_results, "ML Sample": ml_results},
                            caption = "Determinants of Highway Placement - Manual Sample Restriction",
                            label = 'tab:sample_restrict',
                            leaveout = ['owner', 'Intercept' + ''.join([f'City_{c}' for c in ind_sample['city'].unique()[1:]])])

# run lpm with predicted probabilities as covariate on full sample
sample = df.copy()
eps = 1e-6
prob_clipped = sample['prob_hwy'].clip(eps, 1 - eps)
sample['logit'] = np.log(prob_clipped / (1 - prob_clipped))
city_dummies = pd.get_dummies(sample['city'], prefix='city', drop_first=True).values
X = np.column_stack([
        np.ones(len(sample)),
        sample['Residential'].values,
        sample['mblack_1945def'].values,
        sample['Residential'].values * sample['mblack_1945def'].values,
        np.log(sample['valueh'].values),
        np.log(sample['rent'].values),
        sample['owner'].values,
        sample['logit'].values,
        city_dummies
    ])
y = sample['hwy'].values

rng = np.random.default_rng(42)
n = len(sample)
n_bootstraps = 1000

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
ml_results = bootstrap_results_to_namespace(beta_hat, boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner', 'Prob(Hwy)'] + [f'City_{c}' for c in sample['city'].unique()[1:]])
ml_results = format_regression_results(ml_results)
ml_cov_table = export_single_regression(ml_results, caption = "Determinants of Highway Placement - Full Sample with Predicted Probability", label = 'tab:ml_prob_cov', widthmultiplier=0.7, leaveout = ['owner', 'Intercept' + ''.join([f'City_{c}' for c in ind_sample['city'].unique()[1:]])])