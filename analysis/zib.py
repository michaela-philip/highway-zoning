import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm
import glob
import os

df = pd.read_pickle('data/output/sample.pkl')
df['rent']   = df['rent'].replace(0, np.nan)
df['valueh'] = df['valueh'].replace(0, np.nan)
df = df.dropna(subset=['rent', 'valueh']).copy().reset_index(drop=True)

# load CNN probabilities
dataroot = 'cnn/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv_files.sort(key=os.path.getmtime, reverse=True)
logits_df = pd.read_csv(csv_files[0])
logits_df['grid_id'] = logits_df['grid_id'].astype(str)

# convert probabilities to logits
eps = 1e-6
prob_clipped = logits_df['pred_hwy'].clip(eps, 1 - eps)
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
    df['residential'].values,
    df['mblack_1945def'].values,
    df['residential'].values * df['mblack_1945def'].values,
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
def fit_zib_em(X, Z, y, max_iter=200, tol=1e-6):
    n = len(y)
    pi = np.full(n, 0.3)
    p  = np.full(n, y.mean())
    log_lik_old = -np.inf

    for iteration in range(max_iter):

        # E-step
        mix  = pi + (1 - pi) * (1 - p)
        tau  = np.zeros(n)
        tau[y == 0] = pi[y == 0] / np.clip(mix[y == 0], 1e-10, 1)

        # M-step: first stage (gamma)
        gamma_model = sm.GLM(
            tau, Z,
            family=sm.families.Binomial(),
            freq_weights=np.ones(n)
        ).fit(disp=False)
        pi = gamma_model.predict(Z)

        # M-step: second stage (beta)
        candidate_weight = np.where(y == 1, 1.0, 1 - tau)
        beta_model = sm.GLM(
            y, X,
            family=sm.families.Binomial(),
            freq_weights=candidate_weight
        ).fit(disp=False)
        p = beta_model.predict(X)

        # convergence check
        mix = pi + (1 - pi) * (1 - p)
        ll  = (np.sum(np.log(np.clip((1 - pi[y == 1]) * p[y == 1], 1e-10, 1))) +
               np.sum(np.log(np.clip(mix[y == 0], 1e-10, 1))))

        delta = np.abs(ll - log_lik_old)
        log_lik_old = ll

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
    
    # point estimate
    beta_model, gamma_model, pi, p, tau = fit_zib_em(X, Z, y)
    beta_hat = beta_model.params.values
    
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            bm, gm, _, _, _ = fit_zib_em(X[idx], Z.iloc[idx], y[idx])
            boot_coefs.append(bm.params.values)
        except Exception:
            continue
    
    boot_coefs = np.array(boot_coefs)
    se = boot_coefs.std(axis=0)
    ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis=0)
    
    return beta_hat, se, ci_lower, ci_upper, boot_coefs

beta_model, gamma_model, pi, p, tau = fit_zib_em(X, Z, y)
beta_hat, se, ci_lower, ci_upper, boot_coefs = bootstrap_zib_em(X, Z, y)
print("ZIB EM results:")
for i, col in enumerate(['Intercept', 'Residential', 'Black', 'Residential x Black', 'Log(Value)', 'Log(Rent)', 'Owner'] + [f'City_{c}' for c in df['city'].unique()[1:]]):
    print(f"{col}: {beta_hat[i]:.4f} (SE: {se[i]:.4f}, 95% CI: [{ci_lower[i]:.4f}, {ci_upper[i]:.4f}])")