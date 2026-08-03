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


def marginal_effects_table(df, x_vars, columns, beta, boot_coefs=None,
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

        # disparate protection: (Black Non-Res - Black Res) - (White Non-Res - White Res)
        did = (predictions['Black Non-Residential'] - predictions['Black Residential']
               - predictions['White Non-Residential'] + predictions['White Residential'])
        if boot_preds['Black Non-Residential'] is not None:
            boot_did = (boot_preds['Black Non-Residential'] - boot_preds['Black Residential']
                        - boot_preds['White Non-Residential'] + boot_preds['White Residential'])
            se_val = np.std(boot_did)
            p_val = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())
            stars = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
            print(f"{'Disparate protection (DiD)':50} {did:10.4f} {se_val:8.4f} {p_val:8.3f}{stars}")
        else:
            print(f"{'Disparate protection (DiD)':50} {did:10.4f} {'--':>8} {'--':>8}")

        print("\n  'Disparate protection (DiD)' is the difference-in-differences:")
        print("  (Black Non-Res - Black Res) - (White Non-Res - White Res)")
        print("  Negative = residential zoning less protective for Black neighborhoods")

        all_results[sv] = cell_estimates

    return all_results if sweep_var is not None else all_results[None]

def export_marginal_effects_table(results, caption, label, widthmultiplier=0.6, notes=None, column_labels=None):
      """
      Export marginal_effects_table() output as a LaTeX table of contrasts (protection
      effects, racial gap, disparate protection), each cell shown as diff^{stars} over (SE),
      recomputed from the paired bootstrap draws already stored per cell.

      Works uniformly whether `results` is:
        - a single {cell_label: (point, se, boot_draws)} dict (no sweep) -> one column, or
        - a {sweep_value: {cell_label: (point, se, boot_draws)}} dict (sweep_var set)
          -> one column per sweep value.

      column_labels optionally renames sweep-value keys to column headers
      (e.g. {v: f'{v:.2f}' for v in sweep_values}); ignored in the single-spec case.
      """
      def stars(p):
          return '^{***}' if p < 0.01 else '^{**}' if p < 0.05 else '^{*}' if p < 0.10 else ''

      def diff_cell(cell_estimates, a, b):
          pa, _, boota = cell_estimates[a]
          pb, _, bootb = cell_estimates[b]
          diff = pa - pb
          if boota is None or bootb is None:
              return f"{diff:.3f}"
          boot_diff = boota - bootb
          se_val = np.std(boot_diff)
          p_val = 2 * min((boot_diff > 0).mean(), (boot_diff < 0).mean())
          return f"\\makecell[tr]{{{diff:.3f}{stars(p_val)} \\\\ ({se_val:.3f})}}"

      def build_column(cell_estimates):
          rows = {}
          for label, (point, se, _) in cell_estimates.items():
              rows[label] = f"{point:.3f}" if se is None else f"\\makecell[tr]{{{point:.3f} \\\\ ({se:.3f})}}"
          for clabel, (a, b) in CONTRASTS.items():
              rows[clabel] = diff_cell(cell_estimates, a, b)

          p_bnr, _, b_bnr = cell_estimates['Black Non-Residential']
          p_br,  _, b_br  = cell_estimates['Black Residential']
          p_wnr, _, b_wnr = cell_estimates['White Non-Residential']
          p_wr,  _, b_wr  = cell_estimates['White Residential']
          did = p_bnr - p_br - p_wnr + p_wr
          if b_bnr is not None:
              boot_did = b_bnr - b_br - b_wnr + b_wr
              se_val = np.std(boot_did)
              p_val = 2 * min((boot_did > 0).mean(), (boot_did < 0).mean())
              rows['Disparate Protection (Black Protection - White Protection)'] = f"\\makecell[tr]{{{did:.3f}{stars(p_val)} \\\\ ({se_val:.3f})}}"
          else:
              rows['Disparate Protection (Black Protection - White Protection)'] = f"{did:.3f}"
          return rows

      # single-spec values are 3-tuples; sweep values are themselves dicts -- normalize both
      # to "one column per key" so the rest of the function doesn't need to branch again
      is_sweep = all(isinstance(v, dict) for v in results.values())
      if is_sweep:
          columns = {(column_labels or {}).get(k, k): build_column(v) for k, v in results.items()}
      else:
          columns = {'Estimate': build_column(results)}

      row_order = list(next(iter(columns.values())).keys())
      table = pd.DataFrame(columns).reindex(row_order)
      table.index.name = None

      export_table(table, caption, label, widthmultiplier, notes)
