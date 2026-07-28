import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from helpers.latex_formatting import export_multiple_regressions
from analysis.lib.data import load_sample, restrict_to_discretionary, merge_cnn_probs, split_by_candidates, add_cnn_interactions
from analysis.lib.bootstrap import bootstrap_lpm_table
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS,
    build_spec, leaveout_except,
)
from data_code.candidates import candidate_dict

# data_code/create_sample.py:318 defines mblack_1945def as pct_black >= 0.60. Recompute
# the indicator (and its Residential interaction) at alternative thresholds to check
# whether the CORE_VARS results are an artifact of that particular cutoff. pct_black is
# carried in sample.pkl as a fraction in [0, 1], so THRESHOLDS_PCT / 100 mirrors the >= 0.60
# comparison in create_sample.py.
THRESHOLDS_PCT = range(30, 71, 5)
BASELINE_PCT = 60
N_BOOTSTRAPS = 1000

df = load_sample()
df = merge_cnn_probs(df, 'predicted_activation-model1*.csv', dataroot='cnn/')
df = add_cnn_interactions(df)
dir_sample, ind_sample = split_by_candidates(df, candidate_dict)

# x_vars/columns are shared across thresholds -- only the mblack_1945def/ResidentialxBlack
# *values* change below, not which columns exist, so the spec only needs to be built once.
x_vars, columns = build_spec(ind_sample, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS, CNN_LOGIT, LOGIT_INTERACTIONS)
leaveout = leaveout_except(columns, keep=[label for _, label in CORE_VARS])
# beta/se come back as pd.Series indexed by `columns` (so beta['Black'] works below), but
# boot_coefs is a plain (n_bootstraps, k) array with no labels, so it still needs positional lookup.
black_idx = columns.index('Black')
resblack_idx = columns.index('Residential x Black')

print(f"{'Threshold':>10}  {'N (Black)':>10}  {'Black coef':>11}  {'p-value':>8}  "
      f"{'Res x Black coef':>17}  {'p-value':>8}")

threshold_results = {}
for pct in THRESHOLDS_PCT:
    if pct == BASELINE_PCT:
        df_thresh = ind_sample
    else:
        df_thresh = ind_sample.copy()
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
          f"{beta['Black']:>11.4f}  {black_p:>8.3f}  "
          f"{beta['Residential x Black']:>17.4f}  {resblack_p:>8.3f}{tag}")

export_multiple_regressions(
    threshold_results,
    caption='Robustness to Majority-Black Threshold (\\% Black Required for Classification)',
    label='tab:robustness/mblack_threshold',
    leaveout=leaveout,
)

print('\nsaved: tables/robustness/mblack_threshold.tex')
