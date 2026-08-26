import numpy as np
import pandas as pd
from types import SimpleNamespace
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from helpers.latex_formatting import format_regression_results
from scipy.stats import norm

def fit_ppml_conley(df, x_vars, columns, y_var='hwy',
                    cutoff_m=1500, coords=None):
    """
    Fit PPML and compute Conley spatial HAC standard errors.
    Returns a SimpleNamespace compatible with format_regression_results.
    
    Parameters
    ----------
    df       : estimation sample
    x_vars   : raw column names (no intercept)
    columns  : friendly display names (no intercept, matches x_vars)
    y_var    : outcome column
    cutoff_m : Conley distance cutoff in meters
    coords   : (n,2) coordinate array, or None to use df.geometry
    """
    
    y = df[y_var].values
    X = np.column_stack([np.ones(len(df)), df[x_vars].values])
    # X[:,0] is always the intercept — no ambiguity
    
    # --- fit ---
    model = sm.GLM(
        y, X,
        family=sm.families.Poisson(link=sm.families.links.Log())
    ).fit(maxiter=200)
    
    if not model.converged:
        raise RuntimeError("PPML did not converge")
    
    beta     = model.params        # length k+1, index 0 = intercept
    mu       = model.fittedvalues  # exp(Xb)
    resid    = y - mu              # Poisson score residuals
    
    # --- Conley SEs ---
    if coords is None:
        coords = np.column_stack([
            df.geometry.centroid.x.values,
            df.geometry.centroid.y.values
        ])
    
    se, V = _conley_inner(X, mu, resid, coords, cutoff_m)
    
    assert len(beta) == len(columns), (
        f"len(beta)={len(beta)} != len(columns)={len(columns)}"
    )
    
    z    = beta / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))
    
    # pseudo R-squared
    try:
        rsq = model.pseudo_rsquared('cs')
    except Exception:
        rsq = None
    
    return SimpleNamespace(
        params   = pd.Series(beta, index=columns),
        bse      = pd.Series(se,   index=columns),
        pvalues  = pd.Series(pval,      index=columns),
        rsquared = rsq,
        nobs     = float(len(y)),
        V        = V,
        mu         = mu,
        X          = X,
        y          = y,
        model      = model,
    )


def _conley_inner(X, mu, resid, coords, cutoff_m):
    """
    Compute Conley sandwich standard errors.
    Separated so it can be reused for LPM (pass mu=ones) or PPML.
    """
    X      = np.asarray(X)
    mu     = np.asarray(mu)
    resid  = np.asarray(resid)
    coords = np.asarray(coords)
    n, k   = X.shape

    Xe    = X * resid[:, np.newaxis]
    inner = np.zeros((k, k))

    chunk_size = 500
    for i in range(0, n, chunk_size):
        i_end   = min(i + chunk_size, n)
        dists   = cdist(coords[i:i_end], coords, metric='euclidean')
        weights = np.maximum(0, 1 - dists / cutoff_m)
        inner  += Xe[i:i_end].T @ (
            weights[:, :, np.newaxis] * Xe[np.newaxis, :, :]
        ).sum(axis=1)

    # PPML bread: (X'WX)^-1 where W = diag(mu)
    XWX   = X.T @ (mu[:, np.newaxis] * X)
    outer = np.linalg.inv(XWX)

    V  = outer @ inner @ outer
    se = np.sqrt(np.diag(V))
    return se, V