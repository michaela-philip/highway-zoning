import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except,
)

# data_code/create_sample.py:318 defines mblack_1945def as pct_black >= 0.60. Recompute
# the indicator (and its Residential interaction) at alternative thresholds to check
# whether the CORE_VARS results are an artifact of that particular cutoff. pct_black is
# carried in sample.pkl as a fraction in [0, 1], so THRESHOLDS_PCT / 100 mirrors the >= 0.60
# comparison in create_sample.py.
THRESHOLDS_PCT = range(30, 71, 5)
BASELINE_PCT = 60
N_BOOTSTRAPS = 1000

df = load_sample()
df_restricted = restrict_to_discretionary(df)

# x_vars/columns are shared across thresholds -- only the mblack_1945def/ResidentialxBlack
# *values* change below, not which columns exist, so the spec only needs to be built once.
x_vars, columns = build_spec(df_restricted, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)
leaveout = leaveout_except(columns, keep=[label for _, label in CORE_VARS])
black_idx = columns.index('Black')
resblack_idx = columns.index('Residential x Black')

print(f"{'Threshold':>10}  {'N (Black)':>10}  {'Black coef':>11}  {'p-value':>8}  "
      f"{'Res x Black coef':>17}  {'p-value':>8}")

threshold_results = {}
for pct in THRESHOLDS_PCT:
    if pct == BASELINE_PCT:
        df_thresh = df_restricted
    else:
        df_thresh = df_restricted.copy()
        df_thresh['mblack_1945def'] = np.where(df_thresh['pct_black'] >= pct / 100, 1, 0)
        df_thresh['ResidentialxBlack'] = df_thresh['Residential'] * df_thresh['mblack_1945def']

    table, beta, se, boot_coefs = bootstrap_lpm_table(df_thresh, x_vars, columns, n_bootstraps=N_BOOTSTRAPS)
    threshold_results[f'{pct}\\%'] = table

    black_draws = boot_coefs[:, black_idx]
    resblack_draws = boot_coefs[:, resblack_idx]
    black_p = 2 * min((black_draws > 0).mean(), (black_draws < 0).mean())
    resblack_p = 2 * min((resblack_draws > 0).mean(), (resblack_draws < 0).mean())

    tag = '  (baseline)' if pct == BASELINE_PCT else ''
    print(f"{pct:>9}%  {int(df_thresh['mblack_1945def'].sum()):>10}  "
          f"{beta[black_idx]:>11.4f}  {black_p:>8.3f}  "
          f"{beta[resblack_idx]:>17.4f}  {resblack_p:>8.3f}{tag}")

export_multiple_regressions(
    threshold_results,
    caption='Robustness to Majority-Black Threshold (\\% Black Required for mblack\\_1945def = 1)',
    label='tab:mblack_threshold_robustness',
    leaveout=leaveout,
)

print('\nsaved: tables/mblack_threshold_robustness.tex')
