import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.lib.specs import RESIDENTIAL_LABEL, BLACK_LABEL, INTERACTION_LABEL
from helpers.latex_formatting import export_table

# Defaults reproduce the original binary Residential x Black behavior exactly. Pass
# black_values=(low, high) with two representative levels of a CONTINUOUS Black measure
# (e.g. dem_access) -- and matching black_labels, e.g. ('Low Black Access', 'High Black
# Access') -- to evaluate/report the same 4-cell x 2-contrast x DiD structure at chosen
# representative levels instead of literal 0/1. residential_values/residential_labels
# generalize the same way, symmetrically, though the intended use in this project is to
# keep Residential binary (it's a discrete zoning category) and only make Black
# continuous. The interaction term math (residential*black, and any sweep interactions)
# is unaffected either way -- multiplying two arbitrary numbers works the same as
# multiplying two 0/1 flags.
DEFAULT_RESIDENTIAL_VALUES = (0, 1)
DEFAULT_BLACK_VALUES = (0, 1)
DEFAULT_RESIDENTIAL_LABELS = ('Non-Residential', 'Residential')
DEFAULT_BLACK_LABELS = ('White', 'Black')


def _cells_and_contrasts(residential_values, black_values, residential_labels, black_labels):
    """Build the 4 cell (residential_value, black_value) pairs, the 2 within-black-level
    Residential contrasts, and the 4 cell names needed for the DiD statistic -- all keyed
    by residential_labels/black_labels so cell/contrast names read sensibly whether
    black_values/residential_values are 0/1 flags or representative levels of a
    continuous measure."""
    r_lo, r_hi = residential_values
    b_lo, b_hi = black_values
    lbl_r_lo, lbl_r_hi = residential_labels
    lbl_b_lo, lbl_b_hi = black_labels

    cells = {
        f'{lbl_b_lo} {lbl_r_lo}': (r_lo, b_lo),
        f'{lbl_b_lo} {lbl_r_hi}': (r_hi, b_lo),
        f'{lbl_b_hi} {lbl_r_lo}': (r_lo, b_hi),
        f'{lbl_b_hi} {lbl_r_hi}': (r_hi, b_hi),
    }
    contrasts = {
        f'{lbl_b_lo} Protection effect': (f'{lbl_b_lo} {lbl_r_lo}', f'{lbl_b_lo} {lbl_r_hi}'),
        f'{lbl_b_hi} Protection effect': (f'{lbl_b_hi} {lbl_r_lo}', f'{lbl_b_hi} {lbl_r_hi}'),
    }
    # (black-hi @ res-lo, black-hi @ res-hi, black-lo @ res-lo, black-lo @ res-hi) --
    # matches the original did_val = Black[Non-Res] - Black[Res] - White[Non-Res] + White[Res]
    did_cells = (
        f'{lbl_b_hi} {lbl_r_lo}', f'{lbl_b_hi} {lbl_r_hi}',
        f'{lbl_b_lo} {lbl_r_lo}', f'{lbl_b_lo} {lbl_r_hi}',
    )
    return cells, contrasts, did_cells


# --------------------------------------------------------------------------
# shared building blocks
# --------------------------------------------------------------------------

def _cell_vectors(df, x_vars, columns, eval_at='mean',
                   sweep_var=None, sweep_label=None, sweep_value=None,
                   sweep_interactions=None,
                   residential_values=DEFAULT_RESIDENTIAL_VALUES, black_values=DEFAULT_BLACK_VALUES,
                   residential_labels=DEFAULT_RESIDENTIAL_LABELS, black_labels=DEFAULT_BLACK_LABELS):
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
                          y is 0/1. See predicted_outcomes_ame_from_fit.
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

    residential_values/black_values are each a (low, high) pair of the actual numbers
    plugged into the Residential/Black columns for the 4 cells -- (0, 1) by default,
    reproducing the original binary behavior exactly. Pass two representative levels of a
    continuous measure instead (e.g. black_values=(df['dem_access'].quantile(0.1),
    df['dem_access'].quantile(0.9))) to evaluate the same 4-cell structure at chosen
    points along a continuous variable -- the interaction math (residential*black, and
    any sweep interactions below) is unaffected, since multiplying two arbitrary numbers
    works the same as multiplying two 0/1 flags. residential_labels/black_labels name
    those two levels for the cell/contrast labels ('White'/'Black' and 'Non-Residential'/
    'Residential' by default; e.g. ('Low Black Access', 'High Black Access') otherwise).
    """
    row_label, col_label, inter_label = RESIDENTIAL_LABEL, BLACK_LABEL, INTERACTION_LABEL
    cells, _, _ = _cells_and_contrasts(residential_values, black_values, residential_labels, black_labels)
    raw_residential = x_vars[columns[1:].index(row_label)]
    raw_black = x_vars[columns[1:].index(col_label)]

    varying_labels = {row_label, col_label, inter_label}
    if sweep_var is not None:
        if sweep_interactions is None:
            raise ValueError(
                "sweep_var was given but sweep_interactions is None -- pass the "
                "(var, label) triple from specs.sweep_interactions_spec(df, black_key, "
                f"{sweep_var!r}, {sweep_label!r})"
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
        # 'own' means: don't force this axis -- use each row's own real value (only
        # meaningful under eval_at='ame', where every row keeps its actual covariates and
        # is a genuine observation, e.g. when df has already been restricted to a real
        # subgroup like dem_access >= 90th percentile). residential/black can be mixed --
        # one forced, one 'own' -- and the interaction below is still recomputed
        # correctly either way, since a scalar times an array broadcasts per row.
        if residential == 'own':
            assert eval_at == 'ame', "residential='own' only makes sense with eval_at='ame'"
            residential = df[raw_residential].values
        if black == 'own':
            assert eval_at == 'ame', "black='own' only makes sense with eval_at='ame'"
            black = df[raw_black].values
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
    mu = np.exp(eta) if link == 'log' else eta
    return float(np.mean(mu))


def _point_estimates(xs, beta, link, contrasts, did_cells):
    """Point predictions only -- no SE/CI/p-values."""
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}
    cell_estimates = {label: (predictions[label], None, None) for label in xs}
    contrast_results = {clabel: (predictions[a] - predictions[b], None, None)
                         for clabel, (a, b) in contrasts.items()}
    bnr, br, wnr, wr = did_cells
    did_val = predictions[bnr] - predictions[br] - predictions[wnr] + predictions[wr]
    return predictions, cell_estimates, contrast_results, (did_val, None, None)


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
    did_val = predictions[bnr] - predictions[br] - predictions[wnr] + predictions[wr]
    did_se = np.std(boot_did)
    did_p = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _gradient(x, beta, link):
    """Gradient wrt beta of predict(x) = mean_i(link^-1(x_i'beta)) over x's n rows (n=1
    for MEM, n=len(df) for AME):
      LPM (link='identity'): gradient of predict(x_i) = x_i        -> mean_i(x_i)
      PPML (link='log'):     gradient of predict(x_i) = exp(x_i'b) * x_i -> mean_i(...)
    Shared by _delta_estimates and any caller that needs to combine gradients across
    cells/groups computed from different (sub)samples -- e.g. comparing a contrast
    computed on one subgroup of df against the same contrast computed on another."""
    Xv = x.values.astype(float)
    if link == 'log':
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            mu = np.exp(Xv @ np.asarray(beta, dtype=float))
        return (mu[:, None] * Xv).mean(axis=0)
    return Xv.mean(axis=0)


def _se_of(g, cov):
    V = np.asarray(cov)
    return np.sqrt(max(g @ V @ g, 0.0))


def _delta_estimates(xs, beta, cov, link, contrasts, did_cells):
    """SE/CI/p-values via the delta method from a coefficient covariance matrix `cov`
    (full, including the intercept row/column, ordered like `columns`) -- e.g. a
    statsmodels results object's .cov_params(), or the Conley sandwich covariance
    (res.V) from analysis.lib.estimators.fit_ppml_conley. See _gradient for the
    per-cell gradient this is built from.
    """
    predictions = {label: _predict(x, beta, link) for label, x in xs.items()}

    def se_of(g):
        return _se_of(g, cov)

    cell_estimates = {label: (predictions[label], se_of(_gradient(x, beta, link)), None) for label, x in xs.items()}

    def contrast(a, b):
        diff = predictions[a] - predictions[b]
        se = se_of(_gradient(xs[a], beta, link) - _gradient(xs[b], beta, link))
        z = diff / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z)))
        return diff, se, p

    contrast_results = {clabel: contrast(a, b) for clabel, (a, b) in contrasts.items()}

    bnr, br, wnr, wr = did_cells
    did_val = predictions[bnr] - predictions[br] - predictions[wnr] + predictions[wr]
    did_grad = (_gradient(xs[bnr], beta, link) - _gradient(xs[br], beta, link)
                - _gradient(xs[wnr], beta, link) + _gradient(xs[wr], beta, link))
    did_se = se_of(did_grad)
    did_z = did_val / did_se if did_se > 0 else np.nan
    did_p = 2 * (1 - norm.cdf(abs(did_z)))

    return predictions, cell_estimates, contrast_results, (did_val, did_se, did_p)


def _stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def _print_table(sv, sweep_label, eval_at, cell_estimates, contrast_results, did):
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
                        sweep_interactions=None,
                        residential_values=DEFAULT_RESIDENTIAL_VALUES, black_values=DEFAULT_BLACK_VALUES,
                        residential_labels=DEFAULT_RESIDENTIAL_LABELS, black_labels=DEFAULT_BLACK_LABELS,
                        verbose=True):
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

    Set link='log' for a PPML/exponential-mean fit, link='identity' (default) for
    OLS/LPM.

    Optionally sweep a third variable interacted with Residential/Black (e.g. a CNN
    logit/probability) across sweep_values, evaluating every cell/contrast at each value
    -- see _cell_vectors for the sweep_var/sweep_label/sweep_values/sweep_interactions
    arguments.

    residential_values/black_values/residential_labels/black_labels: see
    _cells_and_contrasts / _cell_vectors -- defaults reproduce the original binary
    Residential x Black behavior exactly. Pass e.g. black_values=(low, high) with two
    representative levels of a continuous Black measure (and matching black_labels) to
    get the same 4-cell/2-contrast/DiD structure evaluated at those levels instead.

    Returns {cell label: (point estimate, SE or None, bootstrap draws or None)} etc.
    when not sweeping, or {sweep value: {...that same dict...}} when sweep_var is given.
    """
    assert link in ('identity', 'log')
    assert eval_at in ('mean', 'median', 'ame')
    assert boot_coefs is None or cov is None, "pass at most one of boot_coefs / cov"

    beta = np.asarray(beta)
    sweep_grid = sweep_values if sweep_var is not None else [None]
    _, contrasts, did_cells = _cells_and_contrasts(residential_values, black_values,
                                                     residential_labels, black_labels)

    all_results = {}
    for sv in sweep_grid:
        xs = _cell_vectors(df, x_vars, columns, eval_at=eval_at,
                            sweep_var=sweep_var, sweep_label=sweep_label, sweep_value=sv,
                            sweep_interactions=sweep_interactions,
                            residential_values=residential_values, black_values=black_values,
                            residential_labels=residential_labels, black_labels=black_labels)

        if boot_coefs is not None:
            _, cell_estimates, contrast_results, did = _bootstrap_estimates(xs, beta, boot_coefs, link, contrasts, did_cells)
        elif cov is not None:
            _, cell_estimates, contrast_results, did = _delta_estimates(xs, beta, cov, link, contrasts, did_cells)
        else:
            _, cell_estimates, contrast_results, did = _point_estimates(xs, beta, link, contrasts, did_cells)

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
      - a SimpleNamespace like analysis.lib.estimators.fit_ppml_conley() returns,
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


def predicted_outcomes_by_black_group_from_fit(res, df, x_vars, columns, black_groups,
                                                residential_values=DEFAULT_RESIDENTIAL_VALUES,
                                                residential_labels=DEFAULT_RESIDENTIAL_LABELS,
                                                link='identity', verbose=True):
    """
    "How does the Residential protection effect differ between squares that genuinely
    have low vs. high real Black access" -- using actual SUBGROUPS of df (e.g.
    dem_access >= its 90th percentile), never forcing any row's Black-related columns to
    an externally-chosen number the way predicted_outcomes_from_fit(...,
    black_values=(lo, hi)) does (that forces EVERY row in df to those two exact numbers --
    a fixed-value counterfactual, same mechanism as a fixed CNN-logit sweep, and just as
    capable of extrapolating past the data even though lo/hi are themselves real observed
    values, since the row being forced to them usually isn't the row that actually had
    that value).

    This instead restricts df to each real subgroup and evaluates the Residential effect
    there with Black (and its interaction with Residential) left at each row's own actual
    value throughout -- eval_at='ame' always, via black_values=('own', 'own') internally.

    black_groups: an ordered mapping or list of (label, boolean-mask-into-df) pairs, e.g.
        [('Low Black Access', df['dem_access'] <= lo),
         ('High Black Access', df['dem_access'] >= hi)]

    res needs .params plus .cov_params() or .V (delta method) for SEs -- there's
    currently no bootstrap path here (unlike predicted_outcomes_from_fit).

    Returns {'groups': {label: {'cells': {...}, 'contrast': (diff, se, p)}},
             'did': (val, se, p) or None}. 'did' compares the LAST group's Residential
    contrast to the FIRST's and is only computed when exactly 2 groups are given (a
    single DiD isn't well-defined across more than 2).
    """
    black_groups = list(black_groups.items()) if isinstance(black_groups, dict) else list(black_groups)
    assert len(black_groups) >= 2, "need at least 2 black_groups to compare"

    beta = np.asarray(res.params)
    if hasattr(res, 'cov_params'):
        cov = np.asarray(res.cov_params())
    elif hasattr(res, 'V'):
        cov = np.asarray(res.V)
    else:
        raise AttributeError("res has neither .cov_params() nor .V -- can't get a covariance matrix for delta-method SEs")

    lbl_r_lo, lbl_r_hi = residential_labels

    groups_out = {}
    group_grads = {}
    for glabel, mask in black_groups:
        sub = df[mask]
        if len(sub) == 0:
            raise ValueError(f"black group {glabel!r} matched 0 rows")
        xs = _cell_vectors(sub, x_vars, columns, eval_at='ame',
                            residential_values=residential_values, black_values=('own', 'own'),
                            residential_labels=residential_labels, black_labels=(glabel, glabel))
        key_lo, key_hi = f'{glabel} {lbl_r_lo}', f'{glabel} {lbl_r_hi}'
        predictions = {k: _predict(xs[k], beta, link) for k in (key_lo, key_hi)}
        grad_lo, grad_hi = _gradient(xs[key_lo], beta, link), _gradient(xs[key_hi], beta, link)
        diff = predictions[key_hi] - predictions[key_lo]
        se = _se_of(grad_hi - grad_lo, cov)
        z = diff / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z)))
        cell_estimates = {
            key_lo: (predictions[key_lo], _se_of(grad_lo, cov), None),
            key_hi: (predictions[key_hi], _se_of(grad_hi, cov), None),
        }
        if verbose:
            print("\n" + "=" * 70)
            print(f"PREDICTED OUTCOMES | {glabel} (n={len(sub)})")
            print("(Black held at each observation's own real value; only Residential varied)")
            print("=" * 70)
            print(f"\n{'Neighborhood Type':30} {'Predicted':>12} {'SE':>8} {'95% CI':>20}")
            print("-" * 72)
            for k, (pred, se_k, _) in cell_estimates.items():
                print(f"{k:30} {pred:12.4f} {se_k:8.4f} [{pred - 1.96*se_k:.4f}, {pred + 1.96*se_k:.4f}]")
            print(f"\n{glabel} Protection effect: {diff:10.4f}   SE {se:8.4f}   p={p:.3f}{_stars(p)}")
        groups_out[glabel] = {'cells': cell_estimates, 'contrast': (diff, se, p)}
        group_grads[glabel] = grad_hi - grad_lo

    did = None
    if len(black_groups) == 2:
        (lbl_a, _), (lbl_b, _) = black_groups
        did_val = groups_out[lbl_b]['contrast'][0] - groups_out[lbl_a]['contrast'][0]
        did_se = _se_of(group_grads[lbl_b] - group_grads[lbl_a], cov)
        did_z = did_val / did_se if did_se > 0 else np.nan
        did_p = 2 * (1 - norm.cdf(abs(did_z)))
        did = (did_val, did_se, did_p)
        if verbose:
            print(f"\nDisparate protection ({lbl_b} - {lbl_a}): {did_val:10.4f}   SE {did_se:8.4f}   p={did_p:.3f}{_stars(did_p)}")

    return {'groups': groups_out, 'did': did}


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

    has_interactions = sweep_interactions is not None and len(sweep_interactions) > 0

    results = {}
    for b in bin_id.cat.categories:
        sub = df[bin_id == b]
        if len(sub) == 0:
            continue
        if has_interactions:
            out = predicted_outcomes_from_fit(
                res, sub, x_vars, columns, eval_at='ame',
                sweep_var=sweep_var, sweep_label=sweep_label, sweep_values=['own'],
                sweep_interactions=sweep_interactions, verbose=False, **kwargs,
            )
            cell_estimates, contrast_results, did = out['own']['cells'], out['own']['contrasts'], out['own']['did']
        else:
            out = predicted_outcomes_from_fit(
                res, sub, x_vars, columns, eval_at='ame', verbose=False, **kwargs,
            )
            cell_estimates, contrast_results, did = out['cells'], out['contrasts'], out['did']
        if verbose:
            _print_table(f"{b} (n={len(sub)})", sweep_label, 'ame', cell_estimates, contrast_results, did)
        results[b] = {'cells': cell_estimates, 'contrasts': contrast_results, 'did': did}
    return results


def export_predicted_outcomes_table(results, caption, label,
                                     widthmultiplier=0.6,
                                     notes=None, column_labels=None,
                                     black_labels=DEFAULT_BLACK_LABELS):
    """Export output from predicted_outcomes() / predicted_outcomes_from_fit() --
    {'cells': ..., 'contrasts': ..., 'did': ...}, or {sweep value: {...}} when swept.

    Pass the same black_labels used to generate `results` (see predicted_outcomes) so the
    DiD row reads correctly -- e.g. black_labels=('Low Black Access', 'High Black Access')
    for a continuous Black measure, instead of the 'White'/'Black' default."""
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
