import numpy as np
import pandas as pd
import statsmodels.api as sm
from types import SimpleNamespace

from helpers.latex_formatting import format_regression_results

# All three "table" fit functions below (fit_ols, bootstrap_lpm_table, bootstrap_ppml_table)
# share one call signature and return shape:
#
#   table, beta, se, boot_coefs = fit_fn(df, x_vars, columns, y_var='hwy', ...method kwargs...)
#
#   table      -- LaTeX-ready DataFrame from format_regression_results, straight into
#                 export_single_regression / export_multiple_regressions
#   beta, se   -- pd.Series indexed by `columns` (columns[0] is the intercept), so
#                 e.g. beta['Black'] works the same way regardless of which fit was used
#   boot_coefs -- None for fit_ols (no bootstrap draws); (n_bootstraps, k) array for the
#                 two bootstrap methods, e.g. for marginal_effects_table
#
# This means swapping which method a script uses is a one-line change -- the
# build_spec/leaveout_except/export_* code around it doesn't need to change.


def fit_ols(df, x_vars, columns, y_var='hwy', cluster_var='city'):
    """OLS of y_var on x_vars, clustered by cluster_var. See module docstring for the
    shared (table, beta, se, boot_coefs) return shape."""
    X = df[x_vars].copy()
    X.columns = columns[1:]
    X = sm.add_constant(X)
    raw = sm.OLS(df[y_var], X).fit(cov_type='cluster', cov_kwds={'groups': df[cluster_var]})
    table = format_regression_results(raw)
    beta = pd.Series(raw.params.values, index=columns)
    se = pd.Series(raw.bse.values, index=columns)
    return table, beta, se, None


def bootstrap_lpm(sample, x_vars, n_bootstraps=1000, seed=42, y_var='hwy'):
    """Bootstrap a linear probability model of y_var on x_vars, resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(sample)
    y = sample[y_var].values
    X = np.column_stack([np.ones(n), sample[x_vars].values])

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    boot_coefs = np.empty((n_bootstraps, X.shape[1]))
    for b in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_coefs[b] = np.linalg.lstsq(X[boot_idx], y[boot_idx], rcond=None)[0]

    se = boot_coefs.std(axis=0)
    ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis=0)
    return beta_hat, boot_coefs, se, ci_lower, ci_upper, y, X

# wrapper to convert bootstrap output into a SimpleNamespace that mimics a statsmodels results object
def bootstrap_results_to_namespace(beta_hat, boot_coefs, y, X, col_names):
    """Convert bootstrap output into a SimpleNamespace that mimics a statsmodels results object, so it can be passed to 
    format_regression_results() for LaTeX export. Returns a SimpleNamespace with attributes: params, bse, pvalues, rsquared, nobs."""
    n, k = X.shape

    # Standard errors from bootstrap empirical distribution
    bse = boot_coefs.std(axis=0)

    # Z-scores and two-tailed p-values using normal approximation
    # (standard in bootstrap inference)
    z_scores = beta_hat / bse
    from scipy.stats import norm
    pvalues = 2 * (1 - norm.cdf(np.abs(z_scores)))

    # R-squared
    y_hat = X @ beta_hat
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    rsquared = 1 - ss_res / ss_tot

    results = SimpleNamespace(
        params=pd.Series(beta_hat, index=col_names),
        bse=pd.Series(bse, index=col_names),
        pvalues=pd.Series(pvalues, index=col_names),
        rsquared=rsquared,
        nobs=float(n)
    )
    return results


def bootstrap_lpm_table(sample, x_vars, columns, y_var='hwy', n_bootstraps=1000, seed=42):
    """Bootstrap a LPM and return the shared (table, beta, se, boot_coefs) shape --
    see module docstring. `beta`/`se` are returned as pd.Series so callers besides this
    module (e.g. marginal effects, robustness scripts) can index by friendly label."""
    beta, boot_coefs, se, ci_lower, ci_upper, y, X = bootstrap_lpm(sample, x_vars, n_bootstraps, seed, y_var=y_var)
    namespace = bootstrap_results_to_namespace(beta, boot_coefs, y, X, col_names=columns)
    table = format_regression_results(namespace)
    beta = pd.Series(beta, index=columns)
    se = pd.Series(se, index=columns)
    return table, beta, se, boot_coefs


def bootstrap_ppml(df, x_vars, y_var='hwy', n_bootstraps=500, seed=42, strata_var='city'):
    """Bootstrap a Poisson (PPML) fit of y_var on x_vars, stratified by strata_var so every
    bootstrap draw keeps all cities represented. Mirrors bootstrap_lpm's low-level shape
    (raw arrays plus y/X for the table wrapper below), with the PPML-specific extras
    (ci_lower/ci_upper, full_model) appended at the end."""
    rng = np.random.default_rng(seed)
    n   = len(df)
    y   = df[y_var].values
    X   = sm.add_constant(df[x_vars].values, has_constant='add')

    # point estimate
    full_model = sm.GLM(
        y, X,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(cov_type='HC3', maxiter=200)
    beta_hat = full_model.params.copy()

    print(f"Point estimate converged: {full_model.converged}")
    print(f"Bootstrapping {n_bootstraps} draws...")

    # stratified bootstrap indices — resample within each city
    # guarantees all cities represented in every draw
    if strata_var is not None and strata_var in df.columns:
        strata = df[strata_var].values
        strata_vals = np.unique(strata)
        strata_idx  = {s: np.where(strata == s)[0] for s in strata_vals}
    else:
        strata_idx = None

    boot_coefs = np.full((n_bootstraps, len(beta_hat)), np.nan)
    n_failed   = 0
    n_converged = 0

    for b in range(n_bootstraps):
        if b % 100 == 0 and b > 0:
            print(f"  {b}/{n_bootstraps} | failures: {n_failed}")

        if strata_idx is not None:
            # resample within each city, concatenate
            idx = np.concatenate([
                rng.choice(cidx, size=len(cidx), replace=True)
                for cidx in strata_idx.values()
            ])
        else:
            idx = rng.choice(n, size=n, replace=True)

        y_b = y[idx]
        X_b = X[idx]

        if y_b.sum() < 2:
            n_failed += 1
            continue

        try:
            m = sm.GLM(
                y_b, X_b,
                family=sm.families.Poisson(
                    link=sm.families.links.Log()
                )
            ).fit(maxiter=200, disp=False)

            if m.converged:
                boot_coefs[b] = m.params
                n_converged += 1
            else:
                n_failed += 1
                if n_failed <= 3:
                    print(f"  Draw {b}: did not converge")

        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                print(f"  Draw {b} failed: {type(e).__name__}: {e}")

    print(f"\nBootstrap complete:")
    print(f"  Converged:  {n_converged}/{n_bootstraps}")
    print(f"  Failed:     {n_failed}/{n_bootstraps}")

    valid       = ~np.isnan(boot_coefs).any(axis=1)
    boot_valid  = boot_coefs[valid]

    if len(boot_valid) == 0:
        raise RuntimeError(
            "All bootstrap draws failed. Check x_vars for "
            "separation issues or try removing problematic covariates."
        )

    se       = np.std(boot_valid, axis=0)
    ci_lower = np.percentile(boot_valid, 2.5, axis=0)
    ci_upper = np.percentile(boot_valid, 97.5, axis=0)

    return beta_hat, boot_coefs, se, ci_lower, ci_upper, y, X, full_model


def bootstrap_ppml_table(df, x_vars, columns, y_var='hwy', n_bootstraps=500, seed=42, strata_var='city'):
    """Bootstrap a PPML fit and return the shared (table, beta, se, boot_coefs) shape --
    see module docstring. Mirrors bootstrap_lpm_table's call signature exactly (just add
    strata_var), so a script can switch between the two by changing only the function name.
    Note: the exported table's R-squared is the linear R^2 of the PPML linear index against
    y_var (via bootstrap_results_to_namespace), not a Poisson deviance R^2 -- a rough
    diagnostic only, consistent with how the LPM table's R-squared is computed."""
    beta_hat, boot_coefs, se, ci_lower, ci_upper, y, X, full_model = bootstrap_ppml(
        df, x_vars, y_var=y_var, n_bootstraps=n_bootstraps, seed=seed, strata_var=strata_var
    )
    valid = ~np.isnan(boot_coefs).any(axis=1)
    namespace = bootstrap_results_to_namespace(beta_hat, boot_coefs[valid], y, X, col_names=columns)
    table = format_regression_results(namespace)
    beta = pd.Series(beta_hat, index=columns)
    se = pd.Series(se, index=columns)
    return table, beta, se, boot_coefs


def print_ppml_bootstrap_results(beta_hat, boot_coefs, se,
                                  ci_lower, ci_upper, columns):
    """Print a formatted results table."""
    valid = ~np.isnan(boot_coefs).any(axis=1)
    boot_valid = boot_coefs[valid]

    print(f"\n{'='*70}")
    print("PPML BOOTSTRAP RESULTS")
    print(f"({valid.sum()} valid draws)")
    print(f"{'='*70}")
    print(f"\n{'Variable':40} {'Coef':>8} {'Boot SE':>8} "
          f"{'p-val':>8} {'95% CI':>20}")
    print("-" * 88)

    for i, col in enumerate(columns):
        coef   = beta_hat[i]
        se_val = se[i]
        draws  = boot_valid[:, i]
        p_val  = 2 * min((draws > 0).mean(), (draws < 0).mean())
        stars  = ('***' if p_val < 0.01 else '**' if p_val < 0.05
                  else '*' if p_val < 0.10 else '')
        print(f"{col:40} {coef:8.4f} {se_val:8.4f} "
              f"{p_val:8.3f}{stars:3} "
              f"[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")
