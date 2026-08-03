import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from types import SimpleNamespace

from analysis.lib.data import load_sample, restrict_to_discretionary, split_by_candidates, merge_cnn_probs
from analysis.lib.bootstrap import fit_ols, bootstrap_lpm_table
from analysis.lib.specs import CORE_VARS, HH_CONTROLS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, CNN_LOGIT, build_spec
from data_code.candidates import candidate_dict
from helpers.latex_formatting import export_single_regression, export_table, _wrap_threeparttable, format_regression_results


def balance_test(df_direct, df_indirect, demo_vars, geo_vars, var_labels=None,
                  n_bootstrap=1000, seed=42):
    """
    Compare demographic and geographic characteristics between
    direct and indirect samples to assess whether the difference
    in regression results could be driven by sample composition
    rather than the discretion mechanism.

    Tests:
    1. Raw means comparison with bootstrap SEs
    2. Standardized mean differences
    3. Formal test: does corridor membership predict demographics?
    4. SMD heatmap and means-comparison bar charts (Highway squares)

    demo_vars/geo_vars/var_labels should come from the same analysis.lib.specs blocks
    (CORE_VARS/HOUSING_VARS/GEO_CONTROLS/etc.) used to build the main regression spec, so
    the balance test checks the same variables/transformations the regressions condition
    on rather than a separately-maintained list that can drift out of sync.
    """
    np.random.seed(seed)
    var_labels = var_labels or {}

    all_vars = demo_vars + geo_vars

    # --- add corridor indicator and pool ---
    df_direct   = df_direct.copy()
    df_indirect = df_indirect.copy()
    df_direct['indirect']   = 0
    df_indirect['indirect'] = 1
    df_pool = pd.concat([df_direct, df_indirect], ignore_index=True)

    # =========================================================
    # TABLE 1: means, SDs, SMDs for hwy=1 and hwy=0 separately
    # =========================================================
    results = []

    for var in all_vars:
        for hwy_status, label in [(1, 'Highway'), (0, 'Non-Highway')]:
            d  = df_direct[df_direct['hwy']==hwy_status][var].dropna()
            ind = df_indirect[df_indirect['hwy']==hwy_status][var].dropna()

            if len(d) < 2 or len(ind) < 2:
                continue

            mean_d   = d.mean()
            mean_ind = ind.mean()
            std_d    = d.std()
            std_ind  = ind.std()

            # pooled SD for SMD
            pooled_sd = np.sqrt((std_d**2 + std_ind**2) / 2)
            smd = (mean_ind - mean_d) / pooled_sd if pooled_sd > 0 else np.nan

            # bootstrap SE for difference in means
            boot_diffs = []
            for _ in range(n_bootstrap):
                d_b   = np.random.choice(d.values,   len(d),   replace=True)
                ind_b = np.random.choice(ind.values, len(ind), replace=True)
                boot_diffs.append(ind_b.mean() - d_b.mean())
            boot_se = np.std(boot_diffs)
            diff    = mean_ind - mean_d
            t_stat  = diff / boot_se if boot_se > 0 else np.nan
            p_val   = 2 * min(
                (np.array(boot_diffs) > 0).mean(),
                (np.array(boot_diffs) < 0).mean()
            )

            results.append({
                'variable'    : var,
                'sample'      : label,
                'mean_direct' : mean_d,
                'sd_direct'   : std_d,
                'n_direct'    : len(d),
                'mean_indirect': mean_ind,
                'sd_indirect' : std_ind,
                'n_indirect'  : len(ind),
                'diff'        : diff,
                'boot_se'     : boot_se,
                't_stat'      : t_stat,
                'p_val'       : p_val,
                'smd'         : smd
            })

    results_df = pd.DataFrame(results)

    def export_balance_panels(results_df, var_labels, label, caption, widthmultiplier=0.9, notes=None):
        panels = [('Highway', 'Panel A: Highway Squares'),
                    ('Non-Highway', 'Panel B: Non-Highway Squares')]

        def panel_rows(hwy_label, panel_title):
            sub = results_df[results_df['sample'] == hwy_label].copy()
            stars = np.select(
                [sub['p_val'] < 0.01, sub['p_val'] < 0.05, sub['p_val'] < 0.10],
                ['{***}', '{**}', '{*}'], default=''
            )
            sub['Diff'] = [f"\\makecell[tr]{{{d:.3f}{s} \\\\ ({se:.3f})}}"
                            for d, se, s in zip(sub['diff'], sub['boot_se'], stars)]
            table = sub.assign(Variable=sub['variable'].map(lambda v: var_labels.get(v, v))) \
                        .set_index('Variable')[['mean_direct', 'mean_indirect', 'Diff', 'smd']]
            table.columns = ['Inside Corridor', 'Outside Corridor', 'Diff', 'SMD']
            spanner = pd.DataFrame([[''] * table.shape[1]], columns=table.columns, index=[panel_title])
            return pd.concat([spanner, table])

        combined = pd.concat([panel_rows(hwy, title) for hwy, title in panels])
        combined.index.name = None
    
        num_cols = combined.shape[1]
        col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
        text = combined.style.format(precision=3).to_latex(
            environment='longtable', caption=caption,
            label=f'tab:{label}', hrules=True, column_format=col_format,
        )

        # swap each blank spanner row for a bolded, full-width panel header
        lines = text.split('\n')
        for i, line in enumerate(lines):
            for _, title in panels:
                if line.strip().startswith(title):
                    lines[i] = f'\\multicolumn{{{num_cols + 1}}}{{l}}{{\\textbf{{{title}}}}} \\\\'
        text = '\n'.join(lines)
        text = _wrap_threeparttable(text, widthmultiplier, notes, long=True)
        # text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}') \
        #             .replace('\\end{tabular}', '\\end{tabular*}')
        with open(f'tables/robustness/{label}.tex', 'w') as f:
            f.write(text)
        print(f'saved: tables/robustness/{label}.tex')

    notes = "This table shows a comparison of mean demographic and geographic characteristics for grid squares inside the highway corridor (Column 1) and outside the highway corridor (Column 2)." \
    "Column 3 shows the difference in means between the two samples, with standard errors reported in parenthesis and estimated using a bootstrap procedure with 1,000 draws." \
    "Column 4 shows the standardized mean difference (SMD) between the two samples, calculated as the difference in means divided by the pooled standard deviation." \
    "Panel A contains this comparison for grid squares that are intersected by a highway, while Panel B contains this comparison for grid squares that are not intersected by a highway." \
    " *p<0.10, ** p<0.05, *** p<0.01"
    export_balance_panels(results_df, var_labels, 'balance_test',
                            'Balance Test: Direct vs. Indirect Samples', notes = notes)

    # print formatted table
    print("="*80)
    print("BALANCE TEST: Direct vs Indirect Sample")
    print("="*80)

    for hwy_label in ['Highway', 'Non-Highway']:
        sub = results_df[results_df['sample'] == hwy_label]
        print(f"\n--- {hwy_label} Squares ---")
        print(f"  Direct n={sub['n_direct'].iloc[0] if len(sub)>0 else 'N/A'}, "
              f"Indirect n={sub['n_indirect'].iloc[0] if len(sub)>0 else 'N/A'}")
        print(f"\n  {'Variable':30} {'Inside Corridor':>10} {'Outside Corridor':>10} "
              f"{'Diff':>10} {'SE':>8} {'p-val':>8} {'SMD':>8}")
        print(f"  {'-'*84}")

        for _, row in sub.iterrows():
            stars = ''
            if row['p_val'] < 0.01:  stars = '***'
            elif row['p_val'] < 0.05: stars = '**'
            elif row['p_val'] < 0.10: stars = '*'

            vlab = var_labels.get(row['variable'], row['variable'])
            print(f"  {vlab:30} "
                  f"{row['mean_direct']:10.4f} "
                  f"{row['mean_indirect']:10.4f} "
                  f"{row['diff']:10.4f} "
                  f"{row['boot_se']:8.4f} "
                  f"{row['p_val']:8.3f}{stars:3} "
                  f"{row['smd']:8.3f}")

    # =========================================================
    # TEST 2: does corridor membership predict demographics
    # conditional on geography? -- built/fit the same way as every other
    # regression in this project (build_spec + fit_ols), not a hand-rolled OLS
    # =========================================================
    print("\n" + "="*80)
    print("FALSIFICATION: Does corridor membership predict demographics?")
    print("(conditional on geographic controls — should be NO)")
    print("="*80)

    geo_x_vars, geo_columns = build_spec(df_pool, GEO_CONTROLS, LOG_DIST_HWY, [('indirect', 'Indirect')])

    row_labels, coefs, ses, pvals = [0], [0], [0], [0]
    for demo in demo_vars:
        # _, beta, se, _ = fit_ols(df_pool, geo_x_vars, geo_columns, y_var=demo, cluster_var='city')
        _, beta, se, *_ = bootstrap_lpm_table(df_pool, geo_x_vars, geo_columns,y_var = demo)
        coef, se_val = beta['Indirect'], se['Indirect']
        z = coef / se_val if se_val > 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(z)))

        vlab = var_labels.get(demo, demo)
        row_labels.append(vlab)
        coefs.append(coef)
        ses.append(se_val)
        pvals.append(pval)

        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        print(f"  {vlab:30}: coef={coef:+.4f}  SE={se_val:.4f}  p={pval:.3f}{stars}")
  
    namespace = SimpleNamespace(
      params=pd.Series(coefs, index=row_labels),
      bse=pd.Series(ses, index=row_labels),
      pvalues=pd.Series(pvals, index=row_labels),
      rsquared=np.nan,   # not meaningful here -- see leaveout below
      nobs=len(df_pool),
  )
    balance_table = format_regression_results(namespace)
    notes = "This table shows the results of a falsification test to assess whether location outside of the highway corridor predicts demographic characteristics, conditional on geographic controls. " \
        "Each row corresponds to a separate regression of a demographic variable on the corridor indicator and geographic controls. Standard errors are estimated using a bootstrap procedure with 1,000 draws. " \
        "Insignificant results suggest that, conditional on geographic characteristics, location inside or outside the highway corridor does not predict demographic characteristics.* p<0.10, ** p<0.05, *** p<0.01"
    export_single_regression(
      balance_table,
      caption='Falsification Test: Effect of Corridor Membership on Demographics',
      label='tab:robustness/falsification_test',
      widthmultiplier=0.8,
      leaveout=['R-squared'],
      notes = notes  
  )

    # balance_results = pd.DataFrame(balance_results)
    # balance_results.set_index('Variable', inplace=True)
    # export results to latex table
    
    # num_cols = len(balance_results.columns) - 1
    # caption = 'Balance Test: Effect of Demographics on Corridor Membership'
    # label = 'tab:robustness/balance_test'
    # widthmultiplier = 0.8
    # col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    # text = balance_results.style.format(precision=4).to_latex(position_float = 'centering',
    #             caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    # text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}').replace('\\end{tabular}', '\\end{tabular*}')
    # filename = label.split(':')[-1] + '.tex'
    # with open('tables/' + filename, 'w') as f:
    #     f.write(text)
    # print(f"  Results exported to {filename}")

    print("\n  Interpretation: insignificant coefficients mean corridor")
    print("  membership does not predict demographics conditional on")
    print("  geography — the direct/indirect split is not a demographic split.")

    # =========================================================
    # FIGURE: SMD heatmap for highway squares
    # =========================================================
    hwy_results = results_df[results_df['sample'] == 'Highway'].copy()
    hwy_results['var_label'] = hwy_results['variable'].map(
        lambda x: var_labels.get(x, x)
    )
    hwy_results = hwy_results.set_index('var_label')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # panel 1: SMD bar chart for highway squares
    ax = axes[0]
    smds   = hwy_results['smd'].values
    labels_plot = hwy_results.index.tolist()
    colors = ['tomato' if abs(s) > 0.1 else 'steelblue' for s in smds]

    bars = ax.barh(labels_plot, smds, color=colors, alpha=0.7, edgecolor='none')
    ax.axvline(0,    color='black',  linewidth=0.8, linestyle='-')
    ax.axvline(0.1,  color='gray',   linewidth=1,   linestyle='--', alpha=0.7,
               label='|SMD|=0.1 threshold')
    ax.axvline(-0.1, color='gray',   linewidth=1,   linestyle='--', alpha=0.7)
    ax.set_xlabel('Standardized Mean Difference\n(Indirect − Direct)', fontsize=10)
    ax.set_title('Balance: Highway Squares\nDirect vs Indirect Sample',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # annotate with p-values
    for i, (_, row) in enumerate(hwy_results.iterrows()):
        stars = ''
        if row['p_val'] < 0.01:   stars = '***'
        elif row['p_val'] < 0.05: stars = '**'
        elif row['p_val'] < 0.10: stars = '*'
        if stars:
            ax.text(row['smd'] + 0.01, i, stars, va='center', fontsize=8)

    # panel 2: means comparison for key demographic variables
    ax2 = axes[1]
    x      = np.arange(len(demo_vars))
    width  = 0.35

    direct_means   = [results_df[
        (results_df['variable']==v) &
        (results_df['sample']=='Highway')
    ]['mean_direct'].values[0] if len(results_df[
        (results_df['variable']==v) &
        (results_df['sample']=='Highway')
    ]) > 0 else 0 for v in demo_vars]

    indirect_means = [results_df[
        (results_df['variable']==v) &
        (results_df['sample']=='Highway')
    ]['mean_indirect'].values[0] if len(results_df[
        (results_df['variable']==v) &
        (results_df['sample']=='Highway')
    ]) > 0 else 0 for v in demo_vars]

    ax2.bar(x - width/2, direct_means,   width, label='Direct',
            color='steelblue', alpha=0.7, edgecolor='none')
    ax2.bar(x + width/2, indirect_means, width, label='Indirect',
            color='tomato',    alpha=0.7, edgecolor='none')
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [var_labels.get(v, v) for v in demo_vars],
        rotation=30, ha='right', fontsize=9
    )
    ax2.set_ylabel('Mean', fontsize=10)
    ax2.set_title('Demographic Means: Highway Squares\nDirect vs Indirect',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('tables/balance_test.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('\nsaved: tables/balance_test.png')

    return results_df


df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
dir_sample, ind_sample = split_by_candidates(df, candidate_dict)

# demo_vars/geo_vars/var_labels come straight from the same spec blocks used to build the
# main regression spec (e.g. initial_specif.py/lpm_bootstrap.py's
# build_spec(df, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)), so the
# balance test checks exactly the variables/transformations the regressions condition on.
demo_vars = [v for v, _ in CORE_VARS + HOUSING_VARS ]
geo_vars = [v for v, _ in GEO_CONTROLS + LOG_DIST_HWY + CNN_LOGIT + HH_CONTROLS]
var_labels = dict(CORE_VARS + HOUSING_VARS + HH_CONTROLS + GEO_CONTROLS + LOG_DIST_HWY + CNN_LOGIT)

balance_df = balance_test(
    df_direct   = dir_sample,
    df_indirect = ind_sample,
    demo_vars   = demo_vars,
    geo_vars    = geo_vars,
    var_labels  = var_labels,
    n_bootstrap = 1000,
)