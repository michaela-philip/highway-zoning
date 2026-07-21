import numpy as np
import pandas as pd

from analysis.lib.specs import CORE_VARS


def marginal_effects_table(df, x_vars, columns, beta, boot_coefs, eval_at='mean'):
    """
    Predicted P(hwy=1) for the four Residential x Black cells, holding every other
    regressor in x_vars at its sample mean (or median). Works with any (x_vars,
    columns) pair from analysis.lib.specs.build_spec and any (beta, boot_coefs) pair
    from analysis.lib.bootstrap.bootstrap_lpm[_table] -- the sample, spec, and
    controls can vary freely (e.g. across cities, direct/indirect, CNN-conditional).

    Requires x_vars/columns to include CORE_VARS (Residential, Black, and their
    interaction) -- true for every spec built via build_spec in this project.

    Also reports:
    - Protection effect: Residential vs Non-Residential, within race
    - Racial gap: Black vs White, within Residential/Non-Residential
    - Disparate protection (difference-in-differences)

    Returns {cell label: (point estimate, bootstrap SE, bootstrap draws)}.
    """
    row_var, row_label = CORE_VARS[0]
    col_var, col_label = CORE_VARS[1]
    inter_var, inter_label = CORE_VARS[2]

    # every other regressor gets held fixed at its mean/median, keyed by its friendly label
    other_pairs = [(v, c) for v, c in zip(x_vars, columns[1:])
                   if c not in (row_label, col_label, inter_label)]
    other_raw = [v for v, _ in other_pairs]
    eval_vals = df[other_raw].mean() if eval_at == 'mean' else df[other_raw].median()

    def predict_cell(residential, black, coef_vec):
        x = pd.Series(0.0, index=columns)
        x['Intercept'] = 1.0
        x[row_label] = residential
        x[col_label] = black
        x[inter_label] = residential * black
        for raw, friendly in other_pairs:
            x[friendly] = eval_vals[raw]
        return float(x.values @ coef_vec)

    cells = {
        'White Non-Residential': (0, 0),
        'White Residential': (1, 0),
        'Black Non-Residential': (0, 1),
        'Black Residential': (1, 1),
    }

    beta = np.asarray(beta)
    predictions = {label: predict_cell(res, blk, beta) for label, (res, blk) in cells.items()}

    boot_preds = {label: [] for label in cells}
    for bc in boot_coefs:
        if np.any(np.isnan(bc)):
            continue
        for label, (res, blk) in cells.items():
            boot_preds[label].append(predict_cell(res, blk, bc))
    boot_preds = {label: np.array(v) for label, v in boot_preds.items()}

    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS TABLE")
    print(f"(Other variables held at {'mean' if eval_at == 'mean' else 'median'})")
    print("=" * 70)
    print(f"\n{'Neighborhood Type':30} {'P(Highway)':>12} {'SE':>8} {'95% CI':>20}")
    print("-" * 72)

    cell_estimates = {}
    for label in cells:
        pred = predictions[label]
        boot_arr = boot_preds[label]
        se_val = np.std(boot_arr)
        ci_lo = np.percentile(boot_arr, 2.5)
        ci_hi = np.percentile(boot_arr, 97.5)
        cell_estimates[label] = (pred, se_val, boot_arr)
        print(f"{label:30} {pred:12.4f} {se_val:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    print("\n--- Key Contrasts ---")
    contrasts = {
        'Protection effect (White): Res vs Non-Res': ('White Residential', 'White Non-Residential'),
        'Protection effect (Black): Res vs Non-Res': ('Black Residential', 'Black Non-Residential'),
        'Racial gap (Non-Residential): Black vs White': ('Black Non-Residential', 'White Non-Residential'),
        'Racial gap (Residential): Black vs White': ('Black Residential', 'White Residential'),
        'Disparate protection (DiD)': None,  # special case, computed below
    }

    print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} {'p-val':>8}")
    print("-" * 78)

    for label, pair in contrasts.items():
        if pair is None:
            # DiD: (Black Res - Black Non-Res) - (White Res - White Non-Res)
            boot_diff_arr = (
                boot_preds['Black Residential'] - boot_preds['Black Non-Residential']
                - boot_preds['White Residential'] + boot_preds['White Non-Residential']
            )
            diff = (
                predictions['Black Residential'] - predictions['Black Non-Residential']
                - predictions['White Residential'] + predictions['White Non-Residential']
            )
        else:
            a_label, b_label = pair
            diff = predictions[a_label] - predictions[b_label]
            boot_diff_arr = boot_preds[a_label] - boot_preds[b_label]

        se_val = np.std(boot_diff_arr)
        p_val = 2 * min((boot_diff_arr > 0).mean(), (boot_diff_arr < 0).mean())
        stars = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
        print(f"{label:50} {diff:10.4f} {se_val:8.4f} {p_val:8.3f}{stars}")

    print("\n  'Disparate protection (DiD)' is the difference-in-differences:")
    print("  (Black Res - Black Non-Res) - (White Res - White Non-Res)")
    print("  Negative = residential zoning less protective for Black neighborhoods")

    return cell_estimates
