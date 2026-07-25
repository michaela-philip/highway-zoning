import numpy as np
import statsmodels.api as sm

from helpers.latex_formatting import bootstrap_results_to_namespace, format_regression_results


def bootstrap_lpm(sample, x_vars, n_bootstraps=1000, seed=42):
    """Bootstrap a linear probability model of 'hwy' on x_vars, resampling rows with replacement."""
    rng = np.random.default_rng(seed)
    n = len(sample)
    y = sample['hwy'].values
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


def bootstrap_lpm_table(sample, x_vars, columns, n_bootstraps=1000, seed=42):
    """Bootstrap and return a LaTeX-ready formatted results table plus the raw
    beta/boot_coefs, for callers (e.g. marginal effects) that need them directly."""
    beta, boot_coefs, se, ci_lower, ci_upper, y, X = bootstrap_lpm(sample, x_vars, n_bootstraps, seed)
    namespace = bootstrap_results_to_namespace(beta, boot_coefs, y, X, col_names=columns)
    table = format_regression_results(namespace)
    return table, beta, se, boot_coefs


def bootstrap_ppml(df, y_var, x_vars, n_bootstraps=500, seed=42,
                   strata_var='city'):
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
    columns  = ['const'] + x_vars
    
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
    
    return (beta_hat, boot_coefs, se, 
            ci_lower, ci_upper, columns, full_model)


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
