import pandas as pd
import numpy as np

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

    residential = sample['residential'].values
    black = sample['black'].values
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
    return beta_hat, se, ci_lower, ci_upper

# direct sample
out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (df['grid_id'].isin(candidates))].copy()
    treated = df.loc[(df['city'] == city) & (df['hwy']==1) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
    out_frames.append(treated)
dir_sample = pd.concat(out_frames, ignore_index=True)
dir_beta, dir_se, dir_ci_lower, dir_ci_upper = bootstrap_lpm(dir_sample)
dir_results = format_regression_results(bootstrap_results_to_namespace(dir_beta, boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in sample['city'].unique()[1:]]))
dir_results = format_regression_results(dir_results)

# indirect sample
out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
ind_sample = pd.concat(out_frames, ignore_index=True)
ind_beta, ind_se, ind_ci_lower, ind_ci_upper = bootstrap_lpm(ind_sample)
indir_results = bootstrap_results_to_namespace(ind_beta, boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in sample['city'].unique()[1:]])
indir_results = format_regression_results(indir_results)

# export direct and indirect results together
dir_indir_table = export_multiple_regressions({"Direct Sample": dir_results, "Indirect Sample": indir_results},
                            caption = "Determinants of Highway Placement - Manual Sample Restriction",
                            label = 'tab:dir_indir_results',
                            leaveout = ['owner', 'Intercept' + ''.join([f'City_{c}' for c in sample['city'].unique()[1:]])])

# ML based sample 