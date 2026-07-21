import numpy as np

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
