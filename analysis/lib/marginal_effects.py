import numpy as np
import pandas as pd
from scipy.stats import norm

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
        'Protection effect (White): Non-Res vs Res': ('White Non-Residential', 'White Residential'),
        'Protection effect (Black): Non-Res vs Res': ('Black Non-Residential', 'Black Residential'),
        'Racial gap (Non-Residential): White vs Black': ('White Non-Residential', 'Black Non-Residential'),
        'Racial gap (Residential): White vs Black': ('White Residential', 'Black Residential'),
        'Disparate protection (DiD)': None,  # special case, computed below
    }

    print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} {'p-val':>8}")
    print("-" * 78)

    for label, pair in contrasts.items():
        if pair is None:
            # DiD: (Black Non-Res - Black Res) - (White Non-Res - White Res)
            boot_diff_arr = (
                boot_preds['Black Non-Residential'] - boot_preds['Black Residential']
                - boot_preds['White Non-Residential'] + boot_preds['White Residential']
            )
            diff = (
                predictions['Black Non-Residential'] - predictions['Black Residential']
                - predictions['White Non-Residential'] + predictions['White Residential']
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
    print("  (Black Non-Res - Black Res) - (White Non-Res - White Res)")
    print("  Negative = residential zoning less protective for Black neighborhoods")

    return cell_estimates

def ppml_marginal_effects(fitted_model, df, x_vars,
                           columns=None,
                           eval_at='mean',
                           logit_var=None,      # raw column name, e.g. 'logit_hwy'
                           logit_label=None,    # friendly label in columns, e.g. 'Logit(Hwy|Geo)'
                           logit_eval_values=None,  # values at which to evaluate
                           ):
    
    beta = fitted_model.params
    if columns is None:
        columns = fitted_model.params.index.tolist()
    print(columns)
    col_idx = {c: i for i, c in enumerate(columns)}
    
    # intercept_key = 'const' if 'const' in col_idx else 'Intercept'
    
    # core labels that get varied across cells or swept
    core = {'Residential', 'mblack_1945def', 'ResidentialxBlack'}
    if logit_var is not None and logit_label is not None:
        blk_logit_label    = f'Blackx{logit_label}'
        res_logit_label    = f'Residentialx{logit_label}'
        resblk_logit_label = f'ResidentialxBlackx{logit_label}'
        logit_interaction_labels = {
            logit_label, blk_logit_label,
            res_logit_label, resblk_logit_label
        }
        core |= logit_interaction_labels
    else:
        logit_interaction_labels = set()

    # controls held fixed
    ctrl_cols = [
        (v, c) for v, c in zip(x_vars, columns)
        if c not in core
    ]
    ctrl_vals = (
        df[[v for v, _ in ctrl_cols]].mean()
        if eval_at == 'mean'
        else df[[v for v, _ in ctrl_cols]].median()
    )

    # base vector with controls set, demographics zeroed
    base_x = np.zeros(len(columns))
    # base_x[col_idx[intercept_key]] = 1.0
    for raw, friendly in ctrl_cols:
        if friendly in col_idx:
            base_x[col_idx[friendly]] = ctrl_vals[raw]

    cells = {
        'White Non-Residential': (0, 0),
        'White Residential':     (1, 0),
        'Black Non-Residential': (0, 1),
        'Black Residential':     (1, 1),
    }

    def make_x(residential, black, logit_value=None):
        x = base_x.copy()
        if 'Residential' in col_idx:
            x[col_idx['Residential']]       = residential
        if 'mblack_1945def' in col_idx:
            x[col_idx['mblack_1945def']]    = black
        if 'ResidentialxBlack' in col_idx:
            x[col_idx['ResidentialxBlack']] = residential * black

        if logit_var is not None and logit_value is not None:
            lv = logit_value
            if logit_label in col_idx:
                x[col_idx[logit_label]]          = lv
            if blk_logit_label in col_idx:
                x[col_idx[blk_logit_label]]      = black * lv
            if res_logit_label in col_idx:
                x[col_idx[res_logit_label]]      = residential * lv
            if resblk_logit_label in col_idx:
                x[col_idx[resblk_logit_label]]   = residential * black * lv
        return x

    def predicted_prob(x_vec):
        return np.exp(float(x_vec @ beta))

    def delta_se(x_vec):
        mu  = np.exp(x_vec @ beta)
        var = mu**2 * (x_vec @ fitted_model.cov_params() @ x_vec)
        return np.sqrt(max(var, 0))

    def delta_contrast_se(x_a, x_b):
        mu_a = np.exp(x_a @ beta)
        mu_b = np.exp(x_b @ beta)
        grad = mu_a * x_a - mu_b * x_b
        var  = grad @ fitted_model.cov_params() @ grad
        return np.sqrt(max(var, 0))

    def delta_did_se(x_bnr, x_br, x_wnr, x_wr):
        mu_bnr = np.exp(x_bnr @ beta)
        mu_br  = np.exp(x_br  @ beta)
        mu_wnr = np.exp(x_wnr @ beta)
        mu_wr  = np.exp(x_wr  @ beta)
        grad   = mu_bnr*x_bnr - mu_br*x_br - mu_wnr*x_wnr + mu_wr*x_wr
        var    = grad @ fitted_model.cov_params() @ grad
        return np.sqrt(max(var, 0))

    # evaluation points for logit sweep
    if logit_var is not None:
        if logit_eval_values is None:
            logit_eval_values = [df[logit_var].mean()]
    else:
        logit_eval_values = [None]

    all_results = {}

    for lv in logit_eval_values:
        lv_str = f" | CNN logit = {lv:.3f}" if lv is not None else ""
        print(f"\n{'='*70}")
        print(f"PPML MARGINAL EFFECTS{lv_str}")
        print(f"{'='*70}")
        print(f"\n{'Cell':30} {'P(hwy)':>10} {'SE':>8} "
              f"{'z':>7} {'p-val':>8}")
        print("-" * 65)

        xs    = {label: make_x(res, blk, lv)
                 for label, (res, blk) in cells.items()}
        preds = {label: predicted_prob(x) for label, x in xs.items()}

        cell_results = {}
        for label in cells:
            pred  = preds[label]
            se    = delta_se(xs[label])
            z     = pred / se if se > 0 else np.nan
            pval  = 2 * (1 - norm.cdf(abs(z)))
            stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                     else '*' if pval < 0.10 else '')
            cell_results[label] = (pred, se)
            print(f"{label:30} {pred:10.4f} {se:8.4f} "
                  f"{z:7.2f} {pval:8.3f}{stars}")

        print(f"\n{'Contrast':50} {'Diff':>10} {'SE':>8} "
              f"{'z':>7} {'p-val':>8}")
        print("-" * 80)

        contrasts = {
            'Protection (White): NonRes − Res': (
                'White Non-Residential', 'White Residential'),
            'Protection (Black): NonRes − Res': (
                'Black Non-Residential', 'Black Residential'),
            'Racial gap (NonRes): White − Black': (
                'White Non-Residential', 'Black Non-Residential'),
            'Racial gap (Res): White − Black': (
                'White Residential', 'Black Residential'),
        }

        for clabel, (a, b) in contrasts.items():
            diff  = preds[a] - preds[b]
            se    = delta_contrast_se(xs[a], xs[b])
            z     = diff / se if se > 0 else np.nan
            pval  = 2 * (1 - norm.cdf(abs(z)))
            stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                     else '*' if pval < 0.10 else '')
            print(f"{clabel:50} {diff:10.4f} {se:8.4f} "
                  f"{z:7.2f} {pval:8.3f}{stars}")

        did   = (preds['Black Non-Residential'] - preds['Black Residential']
                 - preds['White Non-Residential'] + preds['White Residential'])
        se    = delta_did_se(xs['Black Non-Residential'],
                              xs['Black Residential'],
                              xs['White Non-Residential'],
                              xs['White Residential'])
        z     = did / se if se > 0 else np.nan
        pval  = 2 * (1 - norm.cdf(abs(z)))
        stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                 else '*' if pval < 0.10 else '')
        print(f"\n{'Disparate protection (DiD)':50} {did:10.4f} {se:8.4f} "
              f"{z:7.2f} {pval:8.3f}{stars}")
        print("  DiD = (Black NonRes − Black Res) − (White NonRes − White Res)")
        print("  Negative = residential zoning less protective for Black areas")

        all_results[lv] = {'predictions': preds, 'xs': xs,
                            'cell_results': cell_results}

    # summary sweep
    if len(logit_eval_values) > 1:
        print(f"\n{'='*70}")
        print("DiD ACROSS CNN LOGIT VALUES")
        print(f"{'='*70}")
        print(f"\n{'CNN Logit':>10} {'DiD':>10} {'SE':>8} "
              f"{'z':>7} {'p-val':>8}")
        print("-" * 50)
        for lv, res in all_results.items():
            preds_lv = res['predictions']
            xs_lv    = res['xs']
            did      = (preds_lv['Black Non-Residential']
                        - preds_lv['Black Residential']
                        - preds_lv['White Non-Residential']
                        + preds_lv['White Residential'])
            se       = delta_did_se(xs_lv['Black Non-Residential'],
                                     xs_lv['Black Residential'],
                                     xs_lv['White Non-Residential'],
                                     xs_lv['White Residential'])
            z        = did / se if se > 0 else np.nan
            pval     = 2 * (1 - norm.cdf(abs(z)))
            stars    = ('***' if pval < 0.01 else '**' if pval < 0.05
                        else '*' if pval < 0.10 else '')
            print(f"{lv:10.3f} {did:10.4f} {se:8.4f} "
                  f"{z:7.2f} {pval:8.3f}{stars}")

    return all_results