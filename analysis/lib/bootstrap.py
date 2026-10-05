import numpy as np
import statsmodels.api as sm

from analysis.lib.fit import Fit

# Same interface as analysis.lib.estimators: fit_fn(df, spec, y_var='hwy', ...) -> Fit.
# Swapping which method a script uses is a one-line change; format_regression_results(fit)
# and predicted_outcomes(fit, ...) work on any of them.


def fit_ols(df, spec, y_var='hwy', cluster_var='city'):
    """OLS of y_var on spec, with cluster_var-clustered covariance."""
    X = spec.design_matrix(df)
    raw = sm.OLS(df[y_var].to_numpy(dtype=float), X).fit(
        cov_type='cluster', cov_kwds={'groups': df[cluster_var]})
    return Fit.from_cov(spec, 'identity', raw.params, raw.cov_params(), raw.nobs,
                        rsquared=raw.rsquared, info={'model': raw})


def bootstrap_lpm(df, spec, y_var='hwy', n_bootstraps=1000, seed=42):
    """Linear probability model of y_var on spec, with SEs from a row bootstrap."""
    rng = np.random.default_rng(seed)
    y = df[y_var].to_numpy(dtype=float)
    X = spec.design_matrix(df)
    n = len(y)

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    boot = np.empty((n_bootstraps, X.shape[1]))
    for b in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        boot[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):  # spurious BLAS warnings, see marginal_effects._cell_stats
        resid = y - X @ beta
    rsq = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return Fit.from_boot(spec, 'identity', beta, boot, n, rsquared=rsq)


def bootstrap_ppml(df, spec, y_var='hwy', n_bootstraps=500, seed=42, strata_var='city'):
    """PPML (Poisson, log link) of y_var on spec, with SEs from a bootstrap stratified by
    strata_var so every draw keeps all cities represented. Draws that fail or don't
    converge are dropped (reported). R^2 is the deviance-based pseudo R^2 of the full fit."""
    rng = np.random.default_rng(seed)
    y = df[y_var].to_numpy(dtype=float)
    X = spec.design_matrix(df)
    n = len(y)
    family = sm.families.Poisson(link=sm.families.links.Log())

    full_model = sm.GLM(y, X, family=family).fit(cov_type='HC3', maxiter=200)
    print(f"Point estimate converged: {full_model.converged}")
    print(f"Bootstrapping {n_bootstraps} draws...")

    if strata_var is not None and strata_var in df.columns:
        strata = df[strata_var].to_numpy()
        strata_idx = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    else:
        strata_idx = None

    boot = np.full((n_bootstraps, X.shape[1]), np.nan)
    n_failed = 0
    for b in range(n_bootstraps):
        if b % 100 == 0 and b > 0:
            print(f"  {b}/{n_bootstraps} | failures: {n_failed}")
        if strata_idx is not None:
            idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in strata_idx])
        else:
            idx = rng.choice(n, size=n, replace=True)
        if y[idx].sum() < 2:
            n_failed += 1
            continue
        try:
            m = sm.GLM(y[idx], X[idx], family=family).fit(maxiter=200, disp=False)
        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                print(f"  Draw {b} failed: {type(e).__name__}: {e}")
            continue
        if m.converged:
            boot[b] = m.params
        else:
            n_failed += 1
            if n_failed <= 3:
                print(f"  Draw {b}: did not converge")

    print(f"\nBootstrap complete: {n_bootstraps - n_failed}/{n_bootstraps} converged")
    if n_failed == n_bootstraps:
        raise RuntimeError("All bootstrap draws failed. Check the spec for separation "
                           "issues or try removing problematic covariates.")

    rsq = 1 - full_model.deviance / full_model.null_deviance
    return Fit.from_boot(spec, 'log', full_model.params, boot, n, rsquared=rsq,
                         r2_label='Pseudo R-squared', info={'model': full_model})
