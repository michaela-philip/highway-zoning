import numpy as np
import pandas as pd

from analysis.lib.specs import CORE_VARS
from helpers.latex_formatting import _wrap_threeparttable, export_table

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


def predicted_outcomes_bootstrapped(df, x_vars, columns, beta, boot_coefs=None,
                            link='identity', eval_at='mean',
                            sweep_var=None, sweep_label=None, sweep_values=None,
                            sweep_interactions=None):
    """
    Predicted outcome for the four Residential x Black cells, holding every other
    regressor in x_vars at its sample mean (or median). Works uniformly with the
    (beta, boot_coefs) pair returned by ANY of analysis.lib.bootstrap's standardized fit
    functions -- fit_ols, bootstrap_lpm_table, bootstrap_ppml_table -- so the sample,
    spec, and fit method can all vary freely: set link='log' for a PPML/exponential-mean
    fit (bootstrap_ppml_table), leave link='identity' (default) for OLS/LPM.

    Requires x_vars/columns to include CORE_VARS (Residential, Black, and their
    interaction) -- true for every spec built via build_spec in this project.

    boot_coefs=None (as returned by fit_ols, which has no bootstrap draws) falls back to
    point-estimate-only cells with no SE/CI/p-values -- pass a bootstrap-based beta/
    boot_coefs (bootstrap_lpm_table/bootstrap_ppml_table for the same spec) to get those.

    Optionally sweep a third variable interacted with Residential/Black (e.g. a CNN
    logit/probability) across sweep_values, evaluating every cell/contrast at each value:
      sweep_var           raw column name of the base (non-interacted) term, e.g. 'logit_hwy'
      sweep_label          its friendly label, e.g. 'CNN Logit'
      sweep_values         list of values to evaluate at
      sweep_interactions   the (var, label) block for its 3 interactions with
                           Black/Residential/Residential x Black -- e.g. specs.LOGIT_INTERACTIONS
                           or specs.PROB_INTERACTIONS, same (var, label) block format as
                           CORE_VARS, in [Blackxsweep, Residentialxsweep, ResidentialxBlackxsweep]
                           order (matching how those blocks are defined in analysis/lib/specs.py)

    Returns {cell label: (point estimate, bootstrap SE or None, bootstrap draws or None)}
    when not sweeping, or {sweep value: {...that same dict...}} when sweep_var is given.
    """
    assert link in ('identity', 'log')
    row_var, row_label = CORE_VARS[0]
    col_var, col_label = CORE_VARS[1]
    inter_var, inter_label = CORE_VARS[2]

    varying_labels = {row_label, col_label, inter_label}
    if sweep_var is not None:
        varying_labels.add(sweep_label)
        varying_labels.update(lbl for _, lbl in sweep_interactions)

    # every other regressor gets held fixed at its mean/median, keyed by its friendly label
    other_pairs = [(v, c) for v, c in zip(x_vars, columns[1:]) if c not in varying_labels]
    other_raw = [v for v, _ in other_pairs]
    eval_vals = df[other_raw].mean() if eval_at == 'mean' else df[other_raw].median()

    def make_x(residential, black, sweep_value=None):
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

    def predict(x, coef_vec):
        z = float(x.values @ coef_vec)
        return np.exp(z) if link == 'log' else z

    beta_arr = np.asarray(beta)
    sweep_grid = sweep_values if sweep_var is not None else [None]

    all_results = {}
    for sv in sweep_grid:
        sv_str = f" | {sweep_label} = {sv:.3f}" if sv is not None else ""
        print("\n" + "=" * 70)
        print(f"MARGINAL EFFECTS TABLE{sv_str}")
        print(f"(Other variables held at {'mean' if eval_at == 'mean' else 'median'})")
        print("=" * 70)
        print(f"\n{'Neighborhood Type':30} {'Predicted':>12} {'SE':>8} {'95% CI':>20}")
        print("-" * 72)

        xs = {label: make_x(res, blk, sv) for label, (res, blk) in CELLS.items()}
        predictions = {label: predict(x, beta_arr) for label, x in xs.items()}

        if boot_coefs is not None:
            boot_preds = {label: [] for label in CELLS}
            for bc in boot_coefs:
                if np.any(np.isnan(bc)):
                    continue
                for label, x in xs.items():
                    boot_preds[label].append(predict(x, bc))
            boot_preds = {label: np.array(v) for label, v in boot_preds.items()}
        else:
            boot_preds = {label: None for label in CELLS}

        cell_estimates = {}
        for label in CELLS:
            pred = predictions[label]
            boot_arr = boot_preds[label]
            if boot_arr is not None:
                se_val = np.std(boot_arr)
                ci_lo, ci_hi = np.percentile(boot_arr, [2.5, 97.5])
                cell_estimates[label] = (pred, se_val, boot_arr)
                print(f"{label:30} {pred:12.4f} {se_val:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
            else:
                cell_estimates[label] = (pred, None, None)
                print(f"{label:30} {pred:12.4f} {'--':>8} {'(no bootstrap draws)':>20}")

        print("\n--- Key Contrasts ---")
        print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} {'p-val':>8}")
        print("-" * 78)

        def contrast(a_label, b_label):
            diff = predictions[a_label] - predictions[b_label]
            if boot_preds[a_label] is not None:
                boot_diff = boot_preds[a_label] - boot_preds[b_label]
                se_val = np.std(boot_diff)
                p_val = 2 * min((boot_diff > 0).mean(), (boot_diff < 0).mean())
            else:
                se_val, p_val = None, None
            return diff, se_val, p_val

        for clabel, (a, b) in CONTRASTS.items():
            diff, se_val, p_val = contrast(a, b)
            if se_val is not None:
                stars = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
                print(f"{clabel:50} {diff:10.4f} {se_val:8.4f} {p_val:8.3f}{stars}")
            else:
                print(f"{clabel:50} {diff:10.4f} {'--':>8} {'--':>8}")

        contrast_results = {}
        for clabel, (a, b) in CONTRASTS.items():
            diff, se_val, p_val = contrast(a, b)
            contrast_results[clabel] = (diff, se_val, p_val)

        did = (predictions['Black Non-Residential'] 
                   - predictions['Black Residential']
                   - predictions['White Non-Residential']
                   + predictions['White Residential'])

        did_se_val, did_p_val = None, None  # initialize before conditional

        if boot_preds['Black Non-Residential'] is not None:
            boot_did = (boot_preds['Black Non-Residential'] - boot_preds['Black Residential']
                        - boot_preds['White Non-Residential'] + boot_preds['White Residential'])
            did_se_val = np.std(boot_did)
            did_p_val  = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())
            stars = '***' if did_p_val < 0.01 else '**' if did_p_val < 0.05 else '*' if did_p_val < 0.10 else ''
            print(f"{'Disparate protection (DiD)':50} {did:10.4f} {did_se_val:8.4f} {did_p_val:8.3f}{stars}")
        else:
            print(f"{'Disparate protection (DiD)':50} {did:10.4f} {'--':>8} {'--':>8}")

        all_results[sv] = {
            'cells'     : cell_estimates,
            'contrasts' : contrast_results,
            'did'       : (did, did_se_val, did_p_val)   # None, None when no bootstrap
        }

    return all_results if sweep_var is not None else all_results[None]

def export_predicted_outcomes_table(results, caption, label,
                                   widthmultiplier=0.6,
                                   notes=None, column_labels=None):
    """
    Export output from either predicted_outcomes_bootstrapped() or
    predicted_outcomes_conley() — both now return the same
    {'cells': ..., 'contrasts': ..., 'did': ...} structure.
    """
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

def predicted_outcomes_conley(df, x_vars, columns, beta_full, V_full,
                                   link='identity', eval_at='mean',
                                   sweep_var=None, sweep_label=None,
                                   sweep_values=None, sweep_interactions=None):
    """
    Predicted outcome for the four Residential x Black cells with Conley
    spatial HAC standard errors via the delta method.

    Mirrors predicted_outcomes_bootstrapped() exactly in interface and output format,
    but replaces bootstrap-based SEs with delta method SEs computed from the
    Conley sandwich covariance matrix V_full.

    Parameters
    ----------
    beta_full : full coefficient vector including intercept (length = len(columns))
    V_full    : full Conley sandwich covariance matrix (len(columns) x len(columns))
                including the intercept row/column -- as returned by fit_ppml_conley
                via res.V_full, or _conley_meat() directly
    All other parameters identical to marginal_effects_table().
    """
    from scipy.stats import norm as _norm
    assert link in ('identity', 'log')

    row_var,   row_label   = CORE_VARS[0]
    col_var,   col_label   = CORE_VARS[1]
    inter_var, inter_label = CORE_VARS[2]

    varying_labels = {row_label, col_label, inter_label}
    if sweep_var is not None:
        varying_labels.add(sweep_label)
        varying_labels.update(lbl for _, lbl in sweep_interactions)

    other_pairs = [(v, c) for v, c in zip(x_vars, columns[1:])
                   if c not in varying_labels]
    other_raw  = [v for v, _ in other_pairs]
    eval_vals  = (df[other_raw].mean() if eval_at == 'mean'
                  else df[other_raw].median())

    beta_arr = np.asarray(beta_full)
    V        = np.asarray(V_full)

    def make_x(residential, black, sweep_value=None):
        x = pd.Series(0.0, index=columns)
        x['Intercept'] = 1.0
        x[row_label]   = residential
        x[col_label]   = black
        x[inter_label] = residential * black
        for raw, friendly in other_pairs:
            x[friendly] = eval_vals[raw]
        if sweep_var is not None and sweep_value is not None:
            x[sweep_label] = sweep_value
            for _, lbl in sweep_interactions:
                tokens  = lbl.split(' x ')
                has_row = row_label in tokens
                has_col = col_label in tokens
                if has_row and has_col:
                    x[lbl] = residential * black * sweep_value
                elif has_row:
                    x[lbl] = residential * sweep_value
                elif has_col:
                    x[lbl] = black * sweep_value
                else:
                    raise ValueError(
                        f"{lbl!r} in sweep_interactions doesn't reference "
                        f"{row_label!r} or {col_label!r}"
                    )
        return x

    def predict(x_vec):
        """Point prediction."""
        z = float(np.asarray(x_vec) @ beta_arr)
        return np.exp(z) if link == 'log' else z

    # add this diagnostic
    for label, (res, blk) in CELLS.items():
        x = make_x(res, blk, sweep_values[0])
        print(f"{label}: Residential={x[row_label]}, Black={x[col_label]}, "
            f"pred={predict(x):.6f}")

    def cell_se(x_vec):
        """
        Delta method SE for a single predicted value.
        LPM:  SE = sqrt(x'Vx)
        PPML: SE = exp(x'b) * sqrt(x'Vx)   [gradient of exp is exp * x]
        """
        xv  = np.asarray(x_vec)
        var = xv @ V @ xv
        se  = np.sqrt(max(var, 0.0))
        if link == 'log':
            se = predict(x_vec) * se
        return se

    def contrast_se(x_a, x_b):
        """
        Delta method SE for predict(x_a) - predict(x_b).
        LPM:  gradient = x_a - x_b
        PPML: gradient = exp(x_a'b)*x_a - exp(x_b'b)*x_b
        """
        xa, xb = np.asarray(x_a), np.asarray(x_b)
        if link == 'log':
            grad = predict(xa) * xa - predict(xb) * xb
        else:
            grad = xa - xb
        var = grad @ V @ grad
        return np.sqrt(max(var, 0.0))

    def did_se(x_bnr, x_br, x_wnr, x_wr):
        """
        Delta method SE for
        (predict(x_bnr) - predict(x_br)) - (predict(x_wnr) - predict(x_wr)).
        """
        if link == 'log':
            grad = (predict(x_bnr) * np.asarray(x_bnr)
                    - predict(x_br)  * np.asarray(x_br)
                    - predict(x_wnr) * np.asarray(x_wnr)
                    + predict(x_wr)  * np.asarray(x_wr))
        else:
            grad = (np.asarray(x_bnr) - np.asarray(x_br)
                    - np.asarray(x_wnr) + np.asarray(x_wr))
        var = grad @ V @ grad
        return np.sqrt(max(var, 0.0))

    def fmt_pval(p):
        stars = ('***' if p < 0.01 else '**' if p < 0.05
                 else '*' if p < 0.10 else '')
        return p, stars

    sweep_grid  = sweep_values if sweep_var is not None else [None]
    all_results = {}

    for sv in sweep_grid:
        sv_str = f" | {sweep_label} = {sv:.3f}" if sv is not None else ""
        print("\n" + "=" * 70)
        print(f"PREDICTED OUTCOMES (Conley SEs){sv_str}")
        print(f"(Other variables held at "
              f"{'mean' if eval_at == 'mean' else 'median'})")
        print("=" * 70)
        print(f"\n{'Neighborhood Type':30} {'Predicted':>12} {'SE':>8} "
              f"{'z':>7} {'p-val':>8} {'95% CI':>20}")
        print("-" * 85)

        xs          = {lbl: make_x(res, blk, sv)
                       for lbl, (res, blk) in CELLS.items()}
        predictions = {lbl: predict(x) for lbl, x in xs.items()}

        cell_estimates = {}
        for lbl in CELLS:
            pred   = predictions[lbl]
            se_val = cell_se(xs[lbl])
            z_val  = pred / se_val if se_val > 0 else np.nan
            p_val  = 2 * (1 - _norm.cdf(abs(z_val)))
            ci_lo  = pred - 1.96 * se_val
            ci_hi  = pred + 1.96 * se_val
            p_str, stars = fmt_pval(p_val)
            # store as (point, se, None) — no bootstrap draws
            cell_estimates[lbl] = (pred, se_val, None)
            print(f"{lbl:30} {pred:12.4f} {se_val:8.4f} "
                  f"{z_val:7.2f} {p_str:8.3f}{stars:3} "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}]")

        print("\n--- Key Contrasts ---")
        print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} "
              f"{'z':>7} {'p-val':>8}")
        print("-" * 85)

        contrast_results = {}
        for clabel, (a, b) in CONTRASTS.items():
            diff   = predictions[a] - predictions[b]
            se_val = contrast_se(xs[a], xs[b])
            z_val  = diff / se_val if se_val > 0 else np.nan
            p_val  = 2 * (1 - _norm.cdf(abs(z_val)))
            contrast_results[clabel] = (diff, se_val, p_val)
            print(f"{clabel:50} {diff:10.4f} {se_val:8.4f} {z_val:7.2f} {p_val:8.3f}")

        did_val = (predictions['Black Non-Residential']
                - predictions['Black Residential']
                - predictions['White Non-Residential']
                + predictions['White Residential'])
        did_se_val = did_se(xs['Black Non-Residential'],
                            xs['Black Residential'],
                            xs['White Non-Residential'],
                            xs['White Residential'])
        did_z   = did_val / did_se_val if did_se_val > 0 else np.nan
        did_p   = 2 * (1 - _norm.cdf(abs(did_z)))
        print(f"{'Disparate Protection':50} "
              f"{did_val:10.4f} {did_se_val:8.4f} {did_z:7.2f} {did_p:8.3f}")

        all_results[sv] = {
            'cells'     : cell_estimates,      # {label: (point, se, None)}
            'contrasts' : contrast_results,    # {label: (diff, se, p_val)}
            'did'       : (did_val, did_se_val, did_p),
        }
    print(f"Contrast results: {contrast_results}")
    print(f"DID: {did_val:.4f} ({did_se_val:.4f})")
    return all_results if sweep_var is not None else all_results[None]