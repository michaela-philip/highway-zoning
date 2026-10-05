from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.lib.specs import Spec


# Every estimator in analysis.lib.estimators / analysis.lib.bootstrap returns a Fit, so
# everything downstream -- format_regression_results(fit) for the coefficient table,
# predicted_outcomes(fit, ...) for predictions -- works the same regardless of which
# estimator produced it. The Fit carries its own Spec (to rebuild X for counterfactuals)
# and link (so predictions can't be made on the wrong scale).

LINKS = ('identity', 'log', 'logit')


@dataclass
class Fit:
    spec: Spec
    link: str
    params: pd.Series                 # indexed by spec.columns
    bse: pd.Series
    pvalues: pd.Series
    nobs: float
    rsquared: float = None
    r2_label: str = None              # None -> 'R-squared' in format_regression_results
    cov: np.ndarray = None            # coefficient covariance -> delta-method SEs
    boot: np.ndarray = None           # (n_draws, k) valid bootstrap draws -> bootstrap SEs
    info: dict = field(default_factory=dict)  # estimator-specific extras (mu, n_iter, ...)

    def __post_init__(self):
        assert self.link in LINKS, f"link must be one of {LINKS}"
        assert len(self.params) == len(self.spec.columns), (
            f"{len(self.params)} coefficients for {len(self.spec.columns)} spec columns")

    @property
    def beta(self):
        return self.params.to_numpy(dtype=float)

    @classmethod
    def from_cov(cls, spec, link, beta, cov, nobs, **kwargs):
        """Fit whose SEs/p-values are the normal approximation from a covariance matrix."""
        cov = np.asarray(cov, dtype=float)
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * (1 - norm.cdf(np.abs(np.asarray(beta) / se)))
        cols = spec.columns
        return cls(spec, link, pd.Series(beta, index=cols), pd.Series(se, index=cols),
                   pd.Series(p, index=cols), float(nobs), cov=cov, **kwargs)

    @classmethod
    def from_boot(cls, spec, link, beta, boot, nobs, **kwargs):
        """Fit whose SEs are the bootstrap standard deviation (failed draws -- rows with
        NaN -- dropped) and p-values the normal approximation to them."""
        boot = np.asarray(boot, dtype=float)
        boot = boot[~np.isnan(boot).any(axis=1)]
        se = boot.std(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            p = 2 * (1 - norm.cdf(np.abs(np.asarray(beta) / se)))
        cols = spec.columns
        return cls(spec, link, pd.Series(beta, index=cols), pd.Series(se, index=cols),
                   pd.Series(p, index=cols), float(nobs), boot=boot, **kwargs)
