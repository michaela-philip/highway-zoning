import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.lib.specs import CORE_VARS
from helpers.latex_formatting import export_table

CELLS = {
    'White Non-Residential': (0, 0),
    'White Residential': (1, 0),
    'Black Non-Residential': (0, 1),
    'Black Residential': (1, 1),
}

CONTRASTS = {
    'White Protection effect': ('White Non-Residential', 'White Residential'),
    'Black Protection effect': ('Black Non-Residential', 'Black Residential')
}


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------

def _cell_vectors(df, x_vars, columns, eval_at='mean',
                   sweep_var=None, sweep_label=None, sweep_value=None,
                   sweep_interactions=None):
    """
    Build the regressor row (pd.Series indexed by `columns`) for each of the four
    Residential x Black cells, holding every other regressor in x_vars at its sample
    mean (or median). Requires x_vars/columns to include CORE_VARS (Residential, Black,
    and their interaction) -- true for every spec built via build_spec in this project.

    Pass sweep_var/sweep_label/sweep_value/sweep_interactions to also set a third
    variable (e.g. a CNN logit/probability) and its interactions with
    Black/Residential/Residential x Black at a given value -- sweep_interactions is the
    (var, label) block for those 3 interactions, e.g. specs.LOGIT_INTERACTIONS or
    specs.PROB_INTERACTIONS, in [Blackxsweep, Residentialxsweep, ResidentialxBlackxsweep]
    order (matching how those blocks are defined in analysis/lib/specs.py).
    """
    row_var, row_label = CORE_VARS[0]
    col_var, col_label = CORE_VARS[1]
    inter_var, inter_label = CORE_VARS[2]

    varying_labels = {row_label, col_label, inter_label}
    if sweep_var is not None:
        varying_labels.add(sweep_label)
        varying_labels.update(lbl for _, lbl in sweep_interactions)

    other_pairs = [(v, c) for v, c in zip(x_vars, columns[1:]) if c not in varying_labels]
    other_raw = [v for v, _ in other_pairs]
    eval_vals = df[other_raw].mean() if eval_at == 'mean' else df[other_raw].median()

    def make_x(residential, black):
        x = pd.Series(0.0, index=columns)
        x['Intercept'] = 1.0
        x[row_label] = residential
        x[col_label] = black
        x[inter_label] = residential * black
        for raw, friendly in other_pairs:
            x[friendly] = eval_vals[raw]
        if sweep_var is not None and sweep_value is not None:
            x[sweep_label] = sweep_value
            for _, lbl in sweep_interactions:
                tokens = lbl.split(' x ')
                has_row, has_col = row_label in tokens, col_label in tokens
                if has_row and has_col:
                    x[lbl] = residential * black * sweep_value
                elif has_row:
                    x[lbl] = residential * sweep_value
                elif has_col:
                    x[lbl] = black * sweep_value
                else:
                    raise ValueError(f"{lbl!r} in sweep_interactions doesn't reference {row_label!r} or {col_label!r}")
        return x

    return {label: make_x(res, blk) for label, (res, blk) in CELLS.items()}


def _predict(x, beta, link):
    z = float(np.asarray(x) @ np.asarray(beta))
    return np.exp(z) if link == 'log' else z


def _point_estimates(xs, beta, link):
    """Point predictions only -- no SE/CI/p-values."""
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}
    cell_estimates = {label: (predictions[label], None, None) for label in xs}
    contrast_results = {clabel: (predictions[a] - predictions[b], None, None)
                         for clabel, (a, b) in CONTRASTS.items()}
    did_val = (predictions['Black Non-Residential'] - predictions['Black Residential']
               - predictions['White Non-Residential'] + predictions['White Residential'])
    return predictions, cell_estimates, contrast_results, (did_val, None, None)


def _bootstrap_estimates(xs, beta, boot_coefs, link):
    """SE/CI/p-values from the empirical bootstrap distribution -- as returned by
    fit_ols/bootstrap_lpm_table/bootstrap_ppml_table in analysis.lib.bootstrap."""
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}

    boot_preds = {label: [] for label in xs}
    for bc in boot_coefs:
        if np.any(np.isnan(bc)):
            continue
        for label, x in xs.items():
            boot_preds[label].append(_predict(x, bc, link))
    boot_preds = {label: np.array(v) for label, v in boot_preds.items()}

    cell_estimates = {
        label: (predictions[label], np.std(boot_preds[label]), boot_preds[label])
        for label in xs
    }

    def contrast(a, b):
        diff = predictions[a] - predictions[b]
        boot_diff = boot_preds[a] - boot_preds[b]
        se = np.std(boot_diff)
        p = 2 * min((boot_diff > 0).mean(), (boot_diff < 0).mean())
        return diff, se, p

    contrast_results = {clabel: contrast(a, b) for clabel, (a, b) in CONTRASTS.items()}

    boot_did = (boot_preds['Black Non-Residential'] - boot_preds['Black Residential']
                - boot_preds['White Non-Residential'] + boot_preds['White Residential'])
    did_val = (predictions['Black Non-Residential'] - predictions['Black Residential']
               - predictions['White Non-Residential'] + predictions['White Residential'])
    did_se = np.std(boot_did)
    did_p = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _delta_estimates(xs, beta, cov, link):
    """SE/CI/p-values via the delta method from a coefficient covariance matrix `cov`
    (full, including the intercept row/column, ordered like `columns`) -- e.g. a
    statsmodels results object's .cov_params(), or the Conley sandwich covariance
    (res.V) from analysis.lib.standard_errors.fit_ppml_conley.

    LPM (link='identity'): gradient of predict(x) = x
    PPML (link='log'):     gradient of predict(x) = exp(x'b) * x
    """
    V = np.asarray(cov)
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}

    def grad(x):
        xv = np.asarray(x)
        return _predict(x, beta, link) * xv if link == 'log' else xv

    def se_of(g):
        return np.sqrt(max(g @ V @ g, 0.0))

    cell_estimates = {label: (predictions[label], se_of(grad(x)), None) for label, x in xs.items()}

    def contrast(a, b):
        diff = predictions[a] - predictions[b]
        se = se_of(grad(xs[a]) - grad(xs[b]))
        z = diff / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z)))
        return diff, se, p

    contrast_results = {clabel: contrast(a, b) for clabel, (a, b) in CONTRASTS.items()}

    did_val = (predictions['Black Non-Residential'] - predictions['Black Residential']
               - predictions['White Non-Residential'] + predictions['White Residential'])
    did_grad = (grad(xs['Black Non-Residential']) - grad(xs['Black Residential'])
                - grad(xs['White Non-Residential']) + grad(xs['White Residential']))
    did_se = se_of(did_grad)
    did_z = did_val / did_se if did_se > 0 else np.nan
    did_p = 2 * (1 - norm.cdf(abs(did_z)))

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def _print_table(sv, sweep_label, eval_at, cell_estimates, contrast_results, did):
    sv_str = f" | {sweep_label} = {sv:.3f}" if sv is not None else ""
    print("\n" + "=" * 70)
    print(f"PREDICTED OUTCOMES{sv_str}")
    print(f"(Other variables held at {'mean' if eval_at == 'mean' else 'median'})")
    print("=" * 70)
    print(f"\n{'Neighborhood Type':30} {'Predicted':>12} {'SE':>8} {'95% CI':>20}")
    print("-" * 72)
    for label, (pred, se, boot) in cell_estimates.items():
        if boot is not None:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            print(f"{label:30} {pred:12.4f} {se:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        elif se is not None:
            print(f"{label:30} {pred:12.4f} {se:8.4f} [{pred - 1.96 * se:.4f}, {pred + 1.96 * se:.4f}]")
        else:
            print(f"{label:30} {pred:12.4f} {'--':>8} {'(no SE available)':>20}")

    print("\n--- Key Contrasts ---")
    print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} {'p-val':>8}")
    print("-" * 78)
    for clabel, (diff, se, p) in contrast_results.items():
        if se is not None:
            print(f"{clabel:50} {diff:10.4f} {se:8.4f} {p:8.3f}{_stars(p)}")
        else:
            print(f"{clabel:50} {diff:10.4f} {'--':>8} {'--':>8}")

    did_val, did_se, did_p = did
    if did_se is not None:
        print(f"{'Disparate protection (DiD)':50} {did_val:10.4f} {did_se:8.4f} {did_p:8.3f}{_stars(did_p)}")
    else:
        print(f"{'Disparate protection (DiD)':50} {did_val:10.4f} {'--':>8} {'--':>8}")


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def predicted_outcomes(df, x_vars, columns, beta, boot_coefs=None, cov=None,
                        link='identity', eval_at='mean',
                        sweep_var=None, sweep_label=None, sweep_values=None,
                        sweep_interactions=None, verbose=True):
    """
    Predicted outcome for the four Residential x Black cells, holding every other
    regressor in x_vars at its sample mean (or median). `beta` is a coefficient vector
    ordered [intercept, *x_vars] to match `columns` -- true for every (beta, ...) pair
    returned by analysis.lib.bootstrap's fit functions, or by predicted_outcomes_from_fit
    below.

    Pass at most one of:
      boot_coefs  (n_bootstraps, k) array of bootstrap draws (fit_ols has none;
                  bootstrap_lpm_table/bootstrap_ppml_table do) -- SEs/CIs/p-values come
                  from the empirical bootstrap distribution.
      cov         full coefficient covariance matrix, e.g. a statsmodels results
                  object's .cov_params(), or the Conley sandwich covariance from
                  analysis.lib.standard_errors.fit_ppml_conley (res.V) -- SEs/CIs/
                  p-values come from the delta method.
    Passing neither returns point estimates only.

    Set link='log' for a PPML/exponential-mean fit, link='identity' (default) for
    OLS/LPM.

    Optionally sweep a third variable interacted with Residential/Black (e.g. a CNN
    logit/probability) across sweep_values, evaluating every cell/contrast at each value
    -- see _cell_vectors for the sweep_var/sweep_label/sweep_values/sweep_interactions
    arguments.

    Returns {cell label: (point estimate, SE or None, bootstrap draws or None)} etc.
    when not sweeping, or {sweep value: {...that same dict...}} when sweep_var is given.
    """
    assert link in ('identity', 'log')
    assert boot_coefs is None or cov is None, "pass at most one of boot_coefs / cov"

    beta = np.asarray(beta)
    sweep_grid = sweep_values if sweep_var is not None else [None]

    all_results = {}
    for sv in sweep_grid:
        xs = _cell_vectors(df, x_vars, columns, eval_at=eval_at,
                            sweep_var=sweep_var, sweep_label=sweep_label, sweep_value=sv,
                            sweep_interactions=sweep_interactions)

        if boot_coefs is not None:
            _, cell_estimates, contrast_results, did = _bootstrap_estimates(xs, beta, boot_coefs, link)
        elif cov is not None:
            _, cell_estimates, contrast_results, did = _delta_estimates(xs, beta, cov, link)
        else:
            _, cell_estimates, contrast_results, did = _point_estimates(xs, beta, link)

        if verbose:
            _print_table(sv, sweep_label, eval_at, cell_estimates, contrast_results, did)

        all_results[sv] = {'cells': cell_estimates, 'contrasts': contrast_results, 'did': did}

    return all_results if sweep_var is not None else all_results[None]


def predicted_outcomes_from_fit(res, df, x_vars, columns, **kwargs):
    """
    predicted_outcomes(), taking a fitted regression result object instead of a bare
    (beta, cov) pair. Covers both:
      - an out-of-the-box statsmodels results object, e.g.
        sm.GLM(y, X, family=sm.families.Poisson(...)).fit() or sm.OLS(y, X).fit(),
        using its .params and .cov_params() for delta-method SEs; or
      - a SimpleNamespace like analysis.lib.standard_errors.fit_ppml_conley() returns,
        using its .params and .V (Conley sandwich covariance) instead.

    res.params must be ordered [intercept, *x_vars] to line up with `columns` -- true
    if X was built as sm.add_constant(df[x_vars]) (the default prepend=True puts the
    constant first), exactly as fit_ppml_conley and analysis.lib.bootstrap's fit
    functions already build it.

    Any remaining kwargs (link, eval_at, sweep_*, verbose) are passed through to
    predicted_outcomes().
    """
    beta = np.asarray(res.params)
    if hasattr(res, 'cov_params'):
        cov = np.asarray(res.cov_params())
    elif hasattr(res, 'V'):
        cov = np.asarray(res.V)
    else:
        raise AttributeError("res has neither .cov_params() nor .V -- can't get a covariance matrix for delta-method SEs")
    return predicted_outcomes(df, x_vars, columns, beta, cov=cov, **kwargs)


def export_predicted_outcomes_table(results, caption, label,
                                     widthmultiplier=0.6,
                                     notes=None, column_labels=None):
    """Export output from predicted_outcomes() / predicted_outcomes_from_fit() --
    {'cells': ..., 'contrasts': ..., 'did': ...}, or {sweep value: {...}} when swept."""
    def stars(p):
        if p is None: return ''
        return ('{***}' if p < 0.01 else '{**}' if p < 0.05
                else '{*}' if p < 0.10 else '')

    def fmt(point, se, p):
        if se is None:
            return f"{point:.3f}"
        return (f"\\makecell[tr]{{{point:.3f}{stars(p)} "
                f"\\\\ ({se:.3f})}}")

    def build_column(sv_results):
        rows = {}

        for lbl, (point, se, _) in sv_results['cells'].items():
            rows[lbl] = fmt(point, se, None)

        for clabel, (diff, se, p) in sv_results['contrasts'].items():
            rows[clabel] = fmt(diff, se, p)

        did_val, did_se, did_p = sv_results['did']
        did_key = 'Disparate Protection (Black Protection - White Protection)'
        rows[did_key] = fmt(did_val, did_se, did_p)

        return rows

    is_sweep = all(
        isinstance(v, dict) and 'cells' in v
        for v in results.values()
    )
    if is_sweep:
        cols = {
            (column_labels or {}).get(k, k): build_column(v)
            for k, v in results.items()
        }
    else:
        cols = {'Estimate': build_column(results)}

    row_order = list(next(iter(cols.values())).keys())
    table = pd.DataFrame(cols).reindex(row_order)
    table.index.name = None
    export_table(table, caption, label, widthmultiplier, notes)
