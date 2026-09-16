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

    # These matmuls spuriously raise divide-by-zero/overflow RuntimeWarnings on some BLAS
    # backends (observed with Accelerate on macOS) whenever a regressor column is all-zero
    # for a chunk/the whole sample, with no actual NaN/Inf in the result -- see
    # analysis/lib/marginal_effects.py's _predict for the same false positive.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
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


def fit_ppml_firth(df, x_vars, columns, y_var='hwy', maxiter=200, tol=1e-8):
    """
    Fit PPML (Poisson, log link) with Firth's bias-reduction correction (Firth 1993,
    generalized to the GLM family by Kosmidis & Firth 2009, "Bias reduction in
    exponential family nonlinear models", Biometrika). Plain Poisson MLE is unstable
    whenever the number of events is small relative to the number of parameters (rule of
    thumb: >=10 events/parameter -- e.g. cnn_specif.py's model has ~80 events over 25
    parameters, ~3.2/parameter): coefficients along any near-separated direction diverge
    toward +-infinity, capped only by the optimizer's iteration limit, which is exactly the
    giant const/city-fixed-effect coefficients seen there. Firth's correction removes the
    O(1/n) bias driving that and keeps coefficients finite even under quasi-complete
    separation, with valid SEs from the corrected information matrix -- no bootstrap
    needed.

    Implementation: for a canonical-link GLM (Poisson-log is canonical), the bias-reduced
    score equations reduce to ordinary IRLS with an adjusted working response
    z*_i = z_i + 0.5*h_i/W_i, where W_i is the usual GLM weight (=mu_i for Poisson-log) and
    h_i = W_i * x_i'(X'WX)^-1 x_i is row i's leverage in the current IRLS weighting --
    otherwise it's exactly standard IRLS. See Kosmidis & Firth (2009) section 3 for the
    canonical-link simplification used here (_firth_irls below).

    Returns a SimpleNamespace compatible with format_regression_results and
    predicted_outcomes_from_fit, same shape as fit_ppml_conley above: .params, .bse,
    .pvalues, .rsquared, .nobs, .V (the Firth-corrected covariance -- model-based, not a
    robust/cluster sandwich; layer one on top separately if you need it).
    """
    y = df[y_var].values.astype(float)
    X = np.column_stack([np.ones(len(df)), df[x_vars].values.astype(float)])

    beta, mu, V, n_iter = _firth_irls(X, y, maxiter=maxiter, tol=tol)

    assert len(beta) == len(columns), (
        f"len(beta)={len(beta)} != len(columns)={len(columns)}"
    )

    se = np.sqrt(np.diag(V))
    z = beta / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    # Deviance-based pseudo R^2 against a null (intercept-only) Poisson fit -- a single
    # parameter estimated from n events is never separation-prone, so the null model
    # doesn't need bias reduction itself.
    null_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson(link=sm.families.links.Log())).fit()
    with np.errstate(divide='ignore', invalid='ignore'):
        dev_terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    deviance = 2 * dev_terms.sum()
    rsq = 1 - deviance / null_model.deviance

    return SimpleNamespace(
        params   = pd.Series(beta, index=columns),
        bse      = pd.Series(se, index=columns),
        pvalues  = pd.Series(pval, index=columns),
        rsquared = rsq,
        nobs     = float(len(y)),
        V        = V,
        mu       = mu,
        X        = X,
        y        = y,
        n_iter   = n_iter,
    )


def fit_ppml_firth_conley(df, x_vars, columns, y_var='hwy', maxiter=200, tol=1e-8,
                           cutoff_m=1500, coords=None):
    """
    Firth-bias-reduced point estimate (fit_ppml_firth) with Conley spatial-HAC SEs
    (fit_ppml_conley's _conley_inner) computed around it, for when you need both: too few
    events/parameter for plain PPML to be stable (fit_ppml_firth's job), AND too few
    clusters (e.g. only a handful of cities) for cluster-robust SEs to be trustworthy, so
    spatial HAC is the right robustness layer instead (fit_ppml_conley's job).

    _conley_inner only needs X, the fitted mu, and the score residuals y-mu -- it doesn't
    care which estimating equation produced them, so this just runs Firth's IRLS for the
    point estimate/mu instead of plain MLE's, then feeds those into the same spatial
    sandwich fit_ppml_conley uses. Caveat: this evaluates the usual (plain-score) sandwich
    formula at the Firth-corrected beta/mu, which is the standard practical approach but
    isn't a from-first-principles derivation of the Firth estimator's own sampling
    variance under spatial dependence (the bias-reduction term's O(1/n) contribution to
    the correction is asymptotically negligible next to the leading-order sandwich
    variance, which is why this approximation is standard, but it is an approximation).

    Returns the same SimpleNamespace shape as fit_ppml_conley/fit_ppml_firth.
    """
    y = df[y_var].values.astype(float)
    X = np.column_stack([np.ones(len(df)), df[x_vars].values.astype(float)])

    beta, mu, _, n_iter = _firth_irls(X, y, maxiter=maxiter, tol=tol)
    resid = y - mu

    if coords is None:
        coords = np.column_stack([
            df.geometry.centroid.x.values,
            df.geometry.centroid.y.values,
        ])

    se, V = _conley_inner(X, mu, resid, coords, cutoff_m)

    assert len(beta) == len(columns), (
        f"len(beta)={len(beta)} != len(columns)={len(columns)}"
    )

    z = beta / se
    pval = 2 * (1 - norm.cdf(np.abs(z)))

    null_model = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Poisson(link=sm.families.links.Log())).fit()
    with np.errstate(divide='ignore', invalid='ignore'):
        dev_terms = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    deviance = 2 * dev_terms.sum()
    rsq = 1 - deviance / null_model.deviance

    return SimpleNamespace(
        params   = pd.Series(beta, index=columns),
        bse      = pd.Series(se, index=columns),
        pvalues  = pd.Series(pval, index=columns),
        rsquared = rsq,
        nobs     = float(len(y)),
        V        = V,
        mu       = mu,
        X        = X,
        y        = y,
        n_iter   = n_iter,
    )


def _firth_irls(X, y, maxiter=200, tol=1e-8):
    """
    IRLS for Poisson-log with Firth's bias-reduction adjustment -- see fit_ppml_firth's
    docstring for the z*_i = z_i + 0.5*h_i/W_i recipe. Returns (beta, mu, V, n_iter), where
    V = (X'WX)^-1 evaluated at the converged beta (the Firth-corrected covariance).

    Convergence is checked on the RELATIVE change in beta (max_j |beta_new_j - beta_j| /
    (|beta_j| + 1)), not the absolute change -- with a rare-events fit some coefficients
    can still be moderately large even after bias reduction, and an absolute tolerance
    tight enough to matter for a near-zero coefficient is unreachable floating-point noise
    for a large one, which is what was cutting fit_ppml_firth's convergence off early.
    """
    n, k = X.shape
    mu = y + 0.1  # standard IRLS start: avoids log(0) for zero counts
    beta = np.zeros(k)

    # (n,k)@(k,) matmuls below spuriously raise divide-by-zero/overflow RuntimeWarnings on
    # some BLAS backends (observed with Accelerate on macOS) with no actual NaN/Inf in the
    # result -- see analysis/lib/marginal_effects.py's _predict for the same false positive.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for n_iter in range(1, maxiter + 1):
            eta = np.log(mu)
            W = mu  # Poisson-log GLM weight: (dmu/deta)^2 / Var(mu) = mu^2/mu = mu
            z = eta + (y - mu) / mu

            XtWX = X.T @ (W[:, None] * X)
            XtWX_inv = np.linalg.inv(XtWX)

            h = W * np.einsum('ij,jk,ik->i', X, XtWX_inv, X)  # weighted hat-matrix diagonal
            z_star = z + 0.5 * h / W

            beta_new = XtWX_inv @ (X.T @ (W * z_star))
            rel_change = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1))

            if rel_change < tol:
                beta = beta_new
                mu = np.exp(X @ beta)
                break

            beta = beta_new
            mu = np.exp(X @ beta)
        else:
            raise RuntimeError(
                f"Firth-corrected PPML did not converge in {maxiter} iterations "
                f"(final relative change {rel_change:.2e}, target {tol:.2e}). "
                f"Final beta: {np.round(beta, 3).tolist()}. "
                "If rel_change is only modestly above tol, try raising maxiter or loosening "
                "tol; if it's bouncing rather than shrinking, that's a real (not just "
                "tolerance) convergence problem worth reporting back with this beta vector."
            )

        W = mu
        V = np.linalg.inv(X.T @ (W[:, None] * X))
    return beta, mu, V, n_iter