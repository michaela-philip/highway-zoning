import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from analysis.lib.specs import RESIDENTIAL_LABEL, BLACK_LABEL, INTERACTION_LABEL
from helpers.latex_formatting import export_table

# Residential is always binary (0/1) here -- it's a discrete zoning category. Black is
# binary (0/1) by DEFAULT, but can be reported at two representative levels of a
# continuous measure instead (e.g. dem_access's 10th/90th percentile) via black_values/
# black_labels below -- standard practice for presenting a continuous regressor's effect,
# as long as the two levels are within the range the model actually saw (real sample
# values, not extrapolated ones) and black_labels makes clear these are representative
# levels, not two literal discrete groups partitioning the sample. If you'd rather Black
# itself be modeled as discrete, threshold it into a 0/1 column before it reaches this
# file instead -- df['Black'] = (df['dem_access'] > x).astype(float) -- exactly like
# every other BLACK_DEFINITIONS entry in specs.py (mblack_40_pct, high_black_share, ...)
# already is a threshold on some underlying continuous measure.
CELLS = {
    'White Non-Residential': (0, 0),
    'White Residential': (1, 0),
    'Black Non-Residential': (0, 1),
    'Black Residential': (1, 1),
}
CONTRASTS = {
    'White Protection effect': ('White Non-Residential', 'White Residential'),
    'Black Protection effect': ('Black Non-Residential', 'Black Residential'),
}
# (Black x Non-Res, Black x Res, White x Non-Res, White x Res) -- the 4 cells the DiD
# statistic combines: did = Black's protection effect minus White's.
DID_CELLS = ('Black Non-Residential', 'Black Residential', 'White Non-Residential', 'White Residential')


def _black_cells(black_values, black_labels):
    """The 4 (residential, black) cell values keyed by name, the 2 Residential contrasts,
    and the DiD cell order -- generalized over black_values/black_labels (Residential
    stays fixed at (0, 1) / 'Non-Residential'/'Residential'). Reduces to CELLS/CONTRASTS/
    DID_CELLS exactly when black_values=(0, 1), black_labels=('White', 'Black') (the
    defaults everywhere this is used)."""
    b_lo, b_hi = black_values
    lbl_lo, lbl_hi = black_labels
    cells = {
        f'{lbl_lo} Non-Residential': (0, b_lo), f'{lbl_lo} Residential': (1, b_lo),
        f'{lbl_hi} Non-Residential': (0, b_hi), f'{lbl_hi} Residential': (1, b_hi),
    }
    contrasts = {
        f'{lbl_lo} Protection effect': (f'{lbl_lo} Non-Residential', f'{lbl_lo} Residential'),
        f'{lbl_hi} Protection effect': (f'{lbl_hi} Non-Residential', f'{lbl_hi} Residential'),
    }
    did_cells = (f'{lbl_hi} Non-Residential', f'{lbl_hi} Residential',
                 f'{lbl_lo} Non-Residential', f'{lbl_lo} Residential')
    return cells, contrasts, did_cells


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------

def _cell_vectors(df, x_vars, columns, eval_at='mean',
                   sweep_var=None, sweep_label=None, sweep_value=None,
                   sweep_interactions=None, black_values=(0, 1), black_labels=('White', 'Black')):
    """
    Build the regressor matrix (pd.DataFrame, columns=`columns`) for each of the four
    Residential x Black cells. Requires columns to include the labels 'Residential',
    'Black', and 'Residential x Black' -- true for every spec built via specs.core_spec()
    in this project, regardless of which BLACK_DEFINITIONS key backed it.

    eval_at controls how every OTHER regressor (not Residential/Black/their interaction,
    nor the sweep block) is set:
      'mean'/'median'  -- MEM: collapse every other regressor to its sample mean/median,
                          giving a single synthetic "average square" row per cell. Cheap,
                          but for a nonlinear link (e.g. link='log'/PPML) E[f(x)] != f(E[x]),
                          and the synthetic row can land far outside the data actually
                          seen by the model, at which point exp(x'beta) can blow up to
                          values that aren't interpretable as probabilities even though
                          y is 0/1. See eval_at='ame' below.
      'ame'            -- average marginal effects: keep every OTHER regressor at each
                          observation's own actual value, predict every row of df under
                          this cell's Residential/Black (and sweep) assignment, then
                          average across the n rows. Every prediction is evaluated at a
                          real covariate combination, so it can't extrapolate the way MEM
                          can. For eval_at='mean' with an identity link the two coincide
                          exactly (linear predictions commute with averaging); they only
                          differ under a nonlinear link like 'log'.

    Pass sweep_var/sweep_label/sweep_value/sweep_interactions whenever x_vars/columns
    include a third variable interacted with Black/Residential/Residential x Black (e.g.
    a CNN logit built via specs.sweep_interactions_spec()) -- WITHOUT this, those
    interaction columns fall into "every other regressor" above and get collapsed to
    their unconditional sample mean/median regardless of the cell's Residential/Black
    values, which is inconsistent (e.g. 'Residential x CNN Logit' held nonzero even in a
    Residential=0 cell) and, under link='log', a common cause of wildly-extrapolated
    predictions. sweep_interactions is the (var, label) block for those 3 interaction
    columns, as returned by specs.sweep_interactions_spec(). sweep_value is either a
    fixed number (e.g. df[sweep_var].mean(), to report a single "at the mean CNN logit"
    column -- the only option under eval_at='mean'/'median', since there's just one row
    to fill in) or the literal string 'own', valid only with eval_at='ame', meaning: use
    each row's own actual sweep_var value instead of one fixed number, so the sweep
    variable and its interactions are held at that row's real value while only
    Residential/Black are counterfactually varied.

    Residential is always binary (0/1). black_values/black_labels let Black be reported
    at two representative levels of a continuous measure instead of literal 0/1 -- see
    the module comment above CELLS for the tradeoffs.
    """
    row_label, col_label, inter_label = RESIDENTIAL_LABEL, BLACK_LABEL, INTERACTION_LABEL
    cells, _, _ = _black_cells(black_values, black_labels)

    varying_labels = {row_label, col_label, inter_label}
    if sweep_var is not None:
        if sweep_label is None:
            raise ValueError("sweep_var was given but sweep_label is None -- pass the friendly "
                              "column label for it (e.g. 'CNN Logit')")
        if sweep_interactions is None:
            raise ValueError(
                "sweep_var was given but sweep_interactions is None -- pass the "
                "(var, label) triple from specs.sweep_interactions_spec(df, black_key, "
                f"{sweep_var!r}, {sweep_label!r}), or [] if sweep_var isn't interacted "
                "with Residential/Black in this model at all"
            )
        varying_labels.add(sweep_label)
        varying_labels.update(lbl for _, lbl in sweep_interactions)

    other_pairs = [(v, c) for v, c in zip(x_vars, columns[1:]) if c not in varying_labels]
    other_raw = [v for v, _ in other_pairs]

    if eval_at == 'ame':
        base = pd.DataFrame(0.0, index=df.index, columns=columns)
        for raw, friendly in other_pairs:
            base[friendly] = df[raw].values
    else:
        eval_vals = df[other_raw].mean() if eval_at == 'mean' else df[other_raw].median()
        base = pd.DataFrame(0.0, index=[0], columns=columns)
        for raw, friendly in other_pairs:
            base[friendly] = eval_vals[raw]
    base['Intercept'] = 1.0

    if sweep_value == 'own':
        assert eval_at == 'ame', "sweep_value='own' only makes sense with eval_at='ame' (MEM has a single row -- pick a fixed sweep_value, e.g. df[sweep_var].mean())"

    def make_x(residential, black):
        x = base.copy()
        x[row_label] = residential
        x[col_label] = black
        x[inter_label] = residential * black
        if sweep_var is not None and sweep_value is not None:
            sv = df[sweep_var].values if sweep_value == 'own' else sweep_value
            x[sweep_label] = sv
            for _, lbl in sweep_interactions:
                tokens = lbl.split(' x ')
                has_row, has_col = row_label in tokens, col_label in tokens
                if has_row and has_col:
                    x[lbl] = residential * black * sv
                elif has_row:
                    x[lbl] = residential * sv
                elif has_col:
                    x[lbl] = black * sv
                else:
                    raise ValueError(f"{lbl!r} in sweep_interactions doesn't reference {row_label!r} or {col_label!r}")
        return x

    return {label: make_x(res, blk) for label, (res, blk) in cells.items()}


def _reference_x(df, x_vars, columns, sweep_var=None, sweep_label=None, sweep_value=None,
                  sweep_interactions=None):
    """
    Regressor matrix using every row's OWN real value for everything, including
    Residential/Black -- no cell is forced. Optionally forces just the sweep variable to
    sweep_value (recomputing its interactions using each row's own real
    Residential/Black), so this can serve as a same-suitability-level reference point
    alongside the 4 (forced) cells -- the 'Sample Average' row from
    predicted_outcomes(..., include_reference_row=True). Always AME-style (real rows),
    regardless of the cells' own eval_at, since the point is an unforced baseline.
    """
    row_label, col_label = RESIDENTIAL_LABEL, BLACK_LABEL
    x = pd.DataFrame(0.0, index=df.index, columns=columns)
    for raw, friendly in zip(x_vars, columns[1:]):
        x[friendly] = df[raw].values
    x['Intercept'] = 1.0

    if sweep_var is not None and sweep_value is not None:
        residential, black = x[row_label].values, x[col_label].values
        x[sweep_label] = sweep_value
        for _, lbl in (sweep_interactions or []):
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


def _predict(X, beta, link):
    """Per-row predictions for a regressor matrix X (DataFrame, n rows), averaged across
    rows. n=1 for the MEM case (eval_at='mean'/'median'), n=len(df) for AME.

    The (n, k) @ (k,) matmul below spuriously raises divide-by-zero/overflow
    RuntimeWarnings on some BLAS backends (observed with Accelerate on macOS) whenever a
    regressor column is all-zero for every row -- e.g. 'Black' in the White cells -- even
    though the actual output has no NaN/Inf; np.errstate suppresses that false positive.
    """
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        eta = X.values.astype(float) @ np.asarray(beta, dtype=float)
    return float(np.mean(_inverse_link(eta, link)))


def _inverse_link(eta, link):
    """Mean function for each supported link: identity (LPM), log (PPML), logit."""
    if link == 'log':
        with np.errstate(over='ignore'):
            return np.exp(eta)
    if link == 'logit':
        return expit(eta)
    return eta


def _did_value(predictions, did_cells):
    """DiD point value from a {label: prediction} dict: Black's protection effect minus
    White's -- did_cells is DID_CELLS, ordered (Black Non-Res, Black Res, White Non-Res,
    White Res). Shared by _point_estimates/_bootstrap_estimates/_delta_estimates, which
    otherwise each repeat this same combination."""
    bnr, br, wnr, wr = did_cells
    return predictions[bnr] - predictions[br] - predictions[wnr] + predictions[wr]


def _point_estimates(xs, beta, link, contrasts, did_cells):
    """Point predictions only -- no SE/CI/p-values."""
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}
    cell_estimates = {label: (predictions[label], None, None) for label in xs}
    contrast_results = {clabel: (predictions[a] - predictions[b], None, None)
                         for clabel, (a, b) in contrasts.items()}
    return predictions, cell_estimates, contrast_results, (_did_value(predictions, did_cells), None, None)


def _bootstrap_estimates(xs, beta, boot_coefs, link, contrasts, did_cells):
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

    contrast_results = {clabel: contrast(a, b) for clabel, (a, b) in contrasts.items()}

    bnr, br, wnr, wr = did_cells
    boot_did = boot_preds[bnr] - boot_preds[br] - boot_preds[wnr] + boot_preds[wr]
    did_val = _did_value(predictions, did_cells)
    did_se = np.std(boot_did)
    did_p = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _gradient(x, beta, link):
    """Gradient wrt beta of predict(x) = mean_i(link^-1(x_i'beta)) over x's n rows (n=1
    for MEM, n=len(df) for AME):
      LPM (link='identity'): gradient of predict(x_i) = x_i        -> mean_i(x_i)
      PPML (link='log'):     gradient of predict(x_i) = exp(x_i'b) * x_i -> mean_i(...)
      Logit (link='logit'):  gradient of predict(x_i) = p_i(1-p_i) * x_i -> mean_i(...)
    Shared by _delta_estimates and any caller that needs to combine gradients across
    cells/groups computed from different (sub)samples -- e.g. comparing a contrast
    computed on one subgroup of df against the same contrast computed on another."""
    Xv = x.values.astype(float)
    if link in ('log', 'logit'):
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            mu = _inverse_link(Xv @ np.asarray(beta, dtype=float), link)
        dmu = mu if link == 'log' else mu * (1 - mu)
        return (dmu[:, None] * Xv).mean(axis=0)
    return Xv.mean(axis=0)


def _se_of(g, cov):
    V = np.asarray(cov)
    return np.sqrt(max(g @ V @ g, 0.0))


def _beta_cov_from_fit(res):
    """(beta, cov) from a fitted result object -- an out-of-the-box statsmodels result
    (.params/.cov_params()) or a SimpleNamespace like analysis.lib.estimators's
    fit_ppml_conley/fit_ppml_firth(_conley) return (.params/.V)."""
    beta = np.asarray(res.params)
    if hasattr(res, 'cov_params'):
        cov = np.asarray(res.cov_params())
    elif hasattr(res, 'V'):
        cov = np.asarray(res.V)
    else:
        raise AttributeError("res has neither .cov_params() nor .V -- can't get a covariance matrix for delta-method SEs")
    return beta, cov


def _delta_contrast(xs, key_a, key_b, beta, cov, link):
    """(diff, se, p) for predictions[key_a] - predictions[key_b] via the delta method
    (note the order: a minus b, matching CONTRASTS' (lo, hi) tuples, e.g. Non-Residential
    minus Residential -- a positive value means the *first* label's predicted rate is
    higher)."""
    diff = _predict(xs[key_a], beta, link) - _predict(xs[key_b], beta, link)
    grad_diff = _gradient(xs[key_a], beta, link) - _gradient(xs[key_b], beta, link)
    se = _se_of(grad_diff, cov)
    z = diff / se if se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z)))
    return diff, se, p


def _delta_estimates(xs, beta, cov, link, contrasts, did_cells):
    """SE/CI/p-values via the delta method from a coefficient covariance matrix `cov`
    (full, including the intercept row/column, ordered like `columns`) -- e.g. a
    statsmodels results object's .cov_params(), or the Conley sandwich covariance
    (res.V) from analysis.lib.estimators.fit_ppml_conley. See _gradient/_delta_contrast
    for the per-cell/per-contrast math this is built from.
    """
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}
    cell_estimates = {label: (predictions[label], _se_of(_gradient(x, beta, link), cov), None)
                       for label, x in xs.items()}

    contrast_results = {clabel: _delta_contrast(xs, a, b, beta, cov, link)
                         for clabel, (a, b) in contrasts.items()}

    bnr, br, wnr, wr = did_cells
    did_val = _did_value(predictions, did_cells)
    did_grad = (_gradient(xs[bnr], beta, link) - _gradient(xs[br], beta, link)
                - _gradient(xs[wnr], beta, link) + _gradient(xs[wr], beta, link))
    did_se = _se_of(did_grad, cov)
    did_z = did_val / did_se if did_se > 0 else np.nan
    did_p = 2 * (1 - norm.cdf(abs(did_z)))

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def _print_table(sv, sweep_label, eval_at, cell_estimates, contrast_results, did=None,
                  did_row_label='Disparate protection (DiD)', link='identity'):
    if sv is None:
        sv_str = ""
    elif sv == 'own':
        sv_str = f" | {sweep_label} = each row's own value"
    elif isinstance(sv, (int, float, np.floating, np.integer)):
        sv_str = f" | {sweep_label} = {sv:.3f}"
    else:
        sv_str = f" | {sweep_label} in {sv}"
    print("\n" + "=" * 70)
    print(f"PREDICTED OUTCOMES{sv_str}")
    eval_desc = {'mean': 'mean', 'median': 'median', 'ame': "each observation's own values (averaged)"}[eval_at]
    print(f"(Other variables held at {eval_desc})")
    print("=" * 70)
    print(f"\n{'Neighborhood Type':30} {'Predicted':>12} {'SE':>8} {'95% CI':>20}")
    print("-" * 72)
    for label, (pred, se, boot) in cell_estimates.items():
        if boot is not None:
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            print(f"{label:30} {pred:12.4f} {se:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        elif se is not None:
            if link == 'log' and pred > 0:
                # Under a log link, a cell prediction is mean_i(exp(x_i'beta)) -- always
                # strictly positive -- but the plain symmetric CI (pred +/- 1.96*se) is a
                # normal approximation on the unbounded real line, and routinely dips
                # below zero when se is large relative to pred (small/thin cells) even
                # though the true quantity can't be negative. Building the interval on
                # log(pred) instead (se_log = se/pred, standard delta-method step) and
                # exponentiating back gives an always-nonnegative, appropriately
                # asymmetric interval. Contrasts/DiD are genuine differences that CAN be
                # negative, so they intentionally keep the plain symmetric interval
                # (see the "Key Contrasts" loop below) -- this only applies to raw cells.
                se_log = se / pred
                with np.errstate(over='ignore'):
                    ci_lo = np.exp(np.log(pred) - 1.96 * se_log)
                    ci_hi = np.exp(np.log(pred) + 1.96 * se_log)
                if not np.isfinite(ci_hi):
                    # degenerate case: pred is so close to 0 that se/pred overflows on
                    # exponentiating back -- fall back to the symmetric interval, still
                    # clipped at 0 since we know that bound is real regardless of link.
                    ci_lo, ci_hi = max(pred - 1.96 * se, 0.0), pred + 1.96 * se
            elif link == 'logit' and 0.0 < pred < 1.0:
                # Same issue, both bounds: a logit-link cell prediction is a probability
                # in (0, 1), but the plain symmetric CI has no knowledge of either wall
                # and routinely dips below 0 or exceeds 1 for thin cells. Delta-method
                # step on the log-odds scale (se_logit = se / (pred*(1-pred))) keeps the
                # interval inside (0, 1) after transforming back with expit.
                se_logit = se / (pred * (1 - pred))
                logit_pred = np.log(pred / (1 - pred))
                with np.errstate(over='ignore'):
                    ci_lo = expit(logit_pred - 1.96 * se_logit)
                    ci_hi = expit(logit_pred + 1.96 * se_logit)
                if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
                    ci_lo, ci_hi = max(pred - 1.96 * se, 0.0), min(pred + 1.96 * se, 1.0)
            else:
                ci_lo, ci_hi = pred - 1.96 * se, pred + 1.96 * se
            print(f"{label:30} {pred:12.4f} {se:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
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

    if did is None:
        return
    did_val, did_se, did_p = did
    if did_se is not None:
        print(f"{did_row_label:50} {did_val:10.4f} {did_se:8.4f} {did_p:8.3f}{_stars(did_p)}")
    else:
        print(f"{did_row_label:50} {did_val:10.4f} {'--':>8} {'--':>8}")


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------

def predicted_outcomes(df, x_vars, columns, beta, boot_coefs=None, cov=None,
                        link='identity', eval_at='mean',
                        sweep_var=None, sweep_label=None, sweep_values=None,
                        sweep_interactions=None,
                        black_values=(0, 1), black_labels=('White', 'Black'),
                        include_reference_row=False, verbose=True):
    """
    Predicted outcome for the four Residential x Black cells. `beta` is a coefficient
    vector ordered [intercept, *x_vars] to match `columns` -- true for every (beta, ...)
    pair returned by analysis.lib.bootstrap's fit functions, or by
    predicted_outcomes_from_fit below.

    eval_at picks how every OTHER regressor is set (see _cell_vectors for the full
    rationale):
      'mean'/'median'  MEM -- collapse to a single synthetic "average square" per cell.
                       Cheap, but under a nonlinear link (link='log'/PPML) this can
                       extrapolate to a covariate combination far from any real
                       observation, producing predictions/SEs that are technically
                       "predicted E[y]" but not sane probabilities even though y is 0/1.
      'ame'            average marginal effects -- predict every row of df under this
                       cell's assignment using that row's own other covariates, then
                       average. Never extrapolates past the data actually used to fit
                       the model; the recommended default when link='log'.

    Pass at most one of:
      boot_coefs  (n_bootstraps, k) array of bootstrap draws (fit_ols has none;
                  bootstrap_lpm_table/bootstrap_ppml_table do) -- SEs/CIs/p-values come
                  from the empirical bootstrap distribution.
      cov         full coefficient covariance matrix, e.g. a statsmodels results
                  object's .cov_params(), or the Conley sandwich covariance from
                  analysis.lib.estimators.fit_ppml_conley (res.V) -- SEs/CIs/
                  p-values come from the delta method.
    Passing neither returns point estimates only.

    Set link='log' for a PPML/exponential-mean fit, link='logit' for a logit fit (e.g.
    analysis.lib.estimators.fit_logit_firth(_conley)), link='identity' (default) for
    OLS/LPM.

    Optionally sweep a third variable interacted with Residential/Black (e.g. a CNN
    logit/probability) across sweep_values, evaluating every cell/contrast at each value
    -- see _cell_vectors for the sweep_var/sweep_label/sweep_values/sweep_interactions
    arguments.

    black_values/black_labels: (0, 1)/('White', 'Black') by default. Pass e.g.
    black_values=(df['dem_access'].quantile(0.1), df['dem_access'].quantile(0.9)),
    black_labels=('Low Black Access', 'High Black Access') to report the same 4-cell
    table at two representative levels of a continuous Black measure instead -- see the
    module comment above CELLS for when that's appropriate.

    include_reference_row=True adds a 'Sample Average' row: the model's average
    predicted rate using every row's own real value for everything, INCLUDING
    Residential/Black (no cell is forced) -- only the sweep variable, if any, is set to
    the current sweep value, same as the 4 cells. Useful alongside the cells as a
    same-suitability-level baseline, e.g. to see whether a given cell's contrast is small
    because that cell sits close to the unconditional average (little going on) or
    because two large, offsetting deviations happen to cancel.

    Returns {cell label: (point estimate, SE or None, bootstrap draws or None)} etc.
    when not sweeping, or {sweep value: {...that same dict...}} when sweep_var is given.
    """
    assert link in ('identity', 'log', 'logit')
    assert eval_at in ('mean', 'median', 'ame')
    assert boot_coefs is None or cov is None, "pass at most one of boot_coefs / cov"

    beta = np.asarray(beta)
    sweep_grid = sweep_values if sweep_var is not None else [None]
    _, contrasts, did_cells = _black_cells(black_values, black_labels)

    all_results = {}
    for sv in sweep_grid:
        xs = _cell_vectors(df, x_vars, columns, eval_at=eval_at,
                            sweep_var=sweep_var, sweep_label=sweep_label, sweep_value=sv,
                            sweep_interactions=sweep_interactions,
                            black_values=black_values, black_labels=black_labels)
        if include_reference_row:
            xs = {**xs, 'Sample Average': _reference_x(df, x_vars, columns, sweep_var=sweep_var,
                                                         sweep_label=sweep_label, sweep_value=sv,
                                                         sweep_interactions=sweep_interactions)}

        if boot_coefs is not None:
            _, cell_estimates, contrast_results, did = _bootstrap_estimates(xs, beta, boot_coefs, link, contrasts, did_cells)
        elif cov is not None:
            _, cell_estimates, contrast_results, did = _delta_estimates(xs, beta, cov, link, contrasts, did_cells)
        else:
            _, cell_estimates, contrast_results, did = _point_estimates(xs, beta, link, contrasts, did_cells)

        if verbose:
            _print_table(sv, sweep_label, eval_at, cell_estimates, contrast_results, did, link=link)

        all_results[sv] = {'cells': cell_estimates, 'contrasts': contrast_results, 'did': did}

    return all_results if sweep_var is not None else all_results[None]


def predicted_outcomes_from_fit(res, df, x_vars, columns, **kwargs):
    """
    predicted_outcomes(), taking a fitted regression result object instead of a bare
    (beta, cov) pair. Covers both:
      - an out-of-the-box statsmodels results object, e.g.
        sm.GLM(y, X, family=sm.families.Poisson(...)).fit() or sm.OLS(y, X).fit(),
        using its .params and .cov_params() for delta-method SEs; or
      - a SimpleNamespace like analysis.lib.estimators.fit_ppml_conley() returns,
        using its .params and .V (Conley sandwich covariance) instead.

    res.params must be ordered [intercept, *x_vars] to line up with `columns` -- true
    if X was built as sm.add_constant(df[x_vars]) (the default prepend=True puts the
    constant first), exactly as fit_ppml_conley and analysis.lib.bootstrap's fit
    functions already build it.

    Any remaining kwargs (link, eval_at, sweep_*, verbose) are passed through to
    predicted_outcomes().
    """
    beta, cov = _beta_cov_from_fit(res)
    return predicted_outcomes(df, x_vars, columns, beta, cov=cov, **kwargs)


def predicted_outcomes_by_stratum_from_fit(res, df, x_vars, columns, sweep_var, bins,
                                            sweep_label=None, sweep_interactions=None,
                                            bin_labels=None, verbose=True, **kwargs):
    """
    "How does the effect vary across levels of `sweep_var`" (e.g. a CNN suitability
    logit), answered WITHOUT ever overriding `sweep_var` to an externally-chosen level.

    predicted_outcomes_from_fit(..., eval_at='ame', sweep_value=<some fixed number>)
    forces every row in df to that one CNN-logit value while keeping every other
    regressor at its own real value -- but that cross of "this row's real city/controls"
    with "an externally chosen suitability level" need not be a combination the model
    ever actually saw, and can extrapolate just as badly as MEM even though every
    individual piece (some row really has that logit value; some row really has those
    controls) was observed somewhere in the data.

    This instead partitions df into bins of `sweep_var` (equal-frequency quantile bins if
    `bins` is an int, explicit edges otherwise -- see pandas.qcut/cut) and runs ordinary
    eval_at='ame', sweep_value='own' (every row keeps ITS OWN sweep_var and interaction
    values; only Residential/Black are counterfactually varied) separately within each
    bin. Every prediction is therefore a real, jointly-observed covariate combination --
    the "how does it vary" comes from which rows are averaged over, not from rewriting
    their covariates.

    Returns {bin: {'cells':..., 'contrasts':..., 'did':...}}, in the same shape
    predicted_outcomes() returns for a sweep -- pass straight to
    export_predicted_outcomes_table(..., column_labels=...).

    sweep_interactions is optional: if `sweep_var` isn't interacted with Residential/Black
    in your model at all (e.g. CNN Logit entered as a plain additive control), pass
    sweep_interactions=None (the default) -- `sweep_var` is then just an ordinary
    covariate, already held at each row's own real value automatically under eval_at='ame'
    with no special-casing needed, and this only uses `sweep_var` to decide which rows
    fall in which bin. Passing sweep_interactions is only for recomputing genuine
    interaction terms (e.g. 'Residential x CNN Logit') consistently with each row's own
    sweep_var value -- see predicted_outcomes()/_cell_vectors for why that recomputation
    matters whenever such terms exist in the model.
    """
    bin_id = pd.qcut(df[sweep_var], bins, labels=bin_labels) if isinstance(bins, int) \
        else pd.cut(df[sweep_var], bins, labels=bin_labels)

    # _cell_vectors only *requires* sweep_interactions when sweep_var is given and it's
    # None (its guard against "you forgot to pass them"); an explicit [] -- "confirmed,
    # sweep_var isn't interacted with anything" -- is already fine, so no branching is
    # needed here for the has-vs-hasn't-got-interactions cases.
    results = {}
    for b in bin_id.cat.categories:
        sub = df[bin_id == b]
        if len(sub) == 0:
            continue
        out = predicted_outcomes_from_fit(
            res, sub, x_vars, columns, eval_at='ame',
            sweep_var=sweep_var, sweep_label=sweep_label, sweep_values=['own'],
            sweep_interactions=sweep_interactions or [], verbose=False, **kwargs,
        )['own']
        if verbose:
            _print_table(f"{b} (n={len(sub)})", sweep_label, 'ame', out['cells'], out['contrasts'], out['did'],
                          link=kwargs.get('link', 'identity'))
        results[b] = out
    return results


def export_predicted_outcomes_table(results, caption, label,
                                     widthmultiplier=0.6,
                                     notes=None, column_labels=None,
                                     black_labels=('White', 'Black')):
    """Export output from predicted_outcomes() / predicted_outcomes_from_fit() --
    {'cells': ..., 'contrasts': ..., 'did': ...}, or {sweep value: {...}} when swept.

    Pass the same black_labels used to generate `results` so the DiD row reads
    correctly -- e.g. black_labels=('Low Black Access', 'High Black Access')."""
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
        did_key = f'Disparate Protection ({black_labels[1]} Protection - {black_labels[0]} Protection)'
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
