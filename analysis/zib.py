import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import os

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results, bootstrap_results_to_namespace

df = pd.read_pickle('data/output/sample.pkl')

df_full = pd.read_pickle('data/output/sample.pkl')
hwy_full = df_full[df_full['hwy'] == 1]
hwy_missing = hwy_full[hwy_full['rent'].isna() | hwy_full['valueh'].isna() | 
                        (hwy_full['rent'] == 0) | (hwy_full['valueh'] == 0)]
print(f"Highway squares dropped due to missing rent/valueh: {len(hwy_missing)}")
print(hwy_missing[['grid_id', 'city', 'rent', 'valueh']].to_string())

df['rent']   = df['rent'].replace(0, np.nan)
df['valueh'] = df['valueh'].replace(0, np.nan)
df = df.dropna(subset=['rent', 'valueh']).copy().reset_index(drop=True)

# load CNN probabilities
dataroot = 'cnn/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv_files.sort(key=os.path.getmtime, reverse=True)
logits_df = pd.read_csv(csv_files[0])
logits_df['grid_id'] = logits_df['grid_id'].astype(str)
df['grid_id'] = df['grid_id'].astype(str)

# convert probabilities to logits
eps = 1e-6
prob_clipped = logits_df['prob_hwy'].clip(eps, 1 - eps)
logits_df['logit'] = np.log(prob_clipped / (1 - prob_clipped))

df = df.merge(logits_df[['grid_id', 'logit']], on='grid_id', how='left')

# minimal pre-filter: remove extreme structural zeros, keep all highways
min_logit = np.log(0.001 / 0.999)
df = df[(df['logit'] > min_logit) | (df['hwy'] == 1)].copy().reset_index(drop=True)
print(f"Estimation sample: {len(df):,} rows, {df['hwy'].sum()} highway squares")

# design matrices
city_dummies = pd.get_dummies(df['city'], drop_first=True).astype(float).values

X = np.column_stack([
    np.ones(len(df)),
    df['Residential'].values,
    df['mblack_1945def'].values,
    df['Residential'].values * df['mblack_1945def'].values,
    np.log(df['valueh'].values),
    np.log(df['rent'].values),
    df['owner'].values,
    city_dummies
])

Z = np.column_stack([
    np.ones(len(df)),
    df['logit'].values
])

y = df['hwy'].values

# --- EM ---
def fit_zib_em(X, Z, y, max_iter=500, tol=1e-4):
    """
    ZIB EM algorithm.
    Increased max_iter and relaxed tol for stability.
    Better initialization from logit.
    """
    from scipy.special import expit
    n = len(y)
    
    # better initialization: use logit to set pi
    # high logit (geographically likely) -> low pi (likely candidate)
    if Z.shape[1] > 1:
        pi = 1 - expit(Z[:, 1])  
    else:
        pi = np.full(n, 0.3)
    pi = np.clip(pi, 0.05, 0.95)
    p  = np.full(n, max(y.mean(), 0.01))
    
    log_lik_old = -np.inf
    ll = -np.inf

    for iteration in range(max_iter):

        # --- E-step ---
        mix = pi + (1 - pi) * (1 - p)
        mix = np.clip(mix, 1e-10, 1)
        tau = np.zeros(n)
        tau[y == 0] = pi[y == 0] / mix[y == 0]
        tau = np.clip(tau, 0, 1)

        # --- M-step: first stage (gamma) ---
        try:
            gamma_model = sm.GLM(
                tau, Z,
                family=sm.families.Binomial(),
                freq_weights=np.ones(n)
            ).fit(disp=False, maxiter=100)
            pi_new = np.clip(gamma_model.predict(Z), 1e-6, 1 - 1e-6)
        except Exception as e:
            print(f"  Gamma GLM failed at iter {iteration}: {e}")
            pi_new = pi  # keep previous if update fails

        # --- M-step: second stage (beta) ---
        candidate_weight = np.where(y == 1, 1.0, 1 - tau)
        candidate_weight = np.clip(candidate_weight, 1e-6, 1)
        try:
            beta_model = sm.GLM(
                y, X,
                family=sm.families.Binomial(),
                freq_weights=candidate_weight
            ).fit(disp=False, maxiter=100)
            p_new = np.clip(beta_model.predict(X), 1e-6, 1 - 1e-6)
        except Exception as e:
            print(f"  Beta GLM failed at iter {iteration}: {e}")
            p_new = p  # keep previous if update fails

        pi = pi_new
        p  = p_new

        # --- convergence check ---
        mix = np.clip(pi + (1 - pi) * (1 - p), 1e-10, 1)
        ll  = (np.sum(np.log(np.clip((1 - pi[y == 1]) * p[y == 1], 1e-10, 1))) +
               np.sum(np.log(mix[y == 0])))

        delta       = np.abs(ll - log_lik_old)
        log_lik_old = ll

        if iteration % 20 == 0:
            print(f"  iter {iteration:3d} | ll={ll:.4f} | delta={delta:.2e} | "
                  f"mean_pi={pi.mean():.3f} | mean_tau={tau[y==0].mean():.3f}")

        if delta < tol:
            print(f"Converged at iteration {iteration+1}, ll={ll:.4f}")
            break
        if iteration == max_iter - 1:
            print(f"Warning: did not converge after {max_iter} iterations, "
                  f"ll={ll:.4f}, delta={delta:.2e}")

    return beta_model, gamma_model, pi, p, tau


def bootstrap_zib_em(X, Z, y, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_coefs = []
    failed = 0

    # point estimate on full sample
    print("Fitting point estimate...")
    beta_model, gamma_model, pi, p, tau = fit_zib_em(X, Z, y)
    beta_hat = np.array(beta_model.params)

    print(f"\nBootstrapping ({n_boot} replications)...")
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        # Z is numpy array so index directly
        try:
            bm, gm, _, _, _ = fit_zib_em(
                X[idx], Z[idx], y[idx], 
                max_iter=200, tol=1e-4
            )
            boot_coefs.append(np.array(bm.params))
        except Exception:
            failed += 1
            continue

        if (b + 1) % 50 == 0:
            print(f"  Bootstrap {b+1}/{n_boot} | failures so far: {failed}")

    if failed > 0:
        print(f"Warning: {failed} bootstrap replications failed and were skipped")

    boot_coefs = np.array(boot_coefs)
    se        = boot_coefs.std(axis=0)
    ci_lower  = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper  = np.percentile(boot_coefs, 97.5, axis=0)

    return beta_hat, se, ci_lower, ci_upper, boot_coefs

beta_model, gamma_model, pi, p, tau = fit_zib_em(X, Z, y)
beta_hat, se, ci_lower, ci_upper, boot_coefs = bootstrap_zib_em(X, Z, y)
print("ZIB EM results:")
for i, col in enumerate(['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in df['city'].unique()[1:]]):
    print(f"{col}: {beta_hat[i]:.4f} (SE: {se[i]:.4f}, 95% CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}])")

zib_results = bootstrap_results_to_namespace(beta_hat, boot_coefs, y, X, col_names = ['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in df['city'].unique()[1:]])
zib_results = format_regression_results(zib_results)
export_single_regression(zib_results, caption = 'Zero-Inflated Binomial Regression Results', label = 'tab:zib_results', widthmultiplier=0.7, leaveout = ['owner', 'Intercept' + ''.join([f'City_{c}' for c in df['city'].unique()])])