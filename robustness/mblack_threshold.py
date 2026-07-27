import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from helpers.latex_formatting import export_multiple_regressions, format_regression_results
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except, fit_ols,
)

# data_code/create_sample.py:318 defines mblack_1945def as pct_black >= 0.60. Recompute
# the indicator (and its Residential interaction) at alternative thresholds to check
# whether the CORE_VARS results are an artifact of that particular cutoff. pct_black is
# carried in sample.pkl as a fraction in [0, 1], so THRESHOLDS_PCT / 100 mirrors the >= 0.60
# comparison in create_sample.py.
THRESHOLDS_PCT = range(30, 71, 5)
BASELINE_PCT = 60


### FUNCTION TO FIT THE MAIN SPECIFICATION ON A GIVEN (ALREADY-RESTRICTED) SAMPLE ###
def fit_spec(df_restricted):
    x_vars, columns = build_spec(df_restricted, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)
    raw = fit_ols(df_restricted, x_vars, columns)
    return format_regression_results(raw), raw, columns


df = load_sample()
df_restricted = restrict_to_discretionary(df)

baseline_results, baseline_raw, columns = fit_spec(df_restricted)
keep_labels = [label for _, label in CORE_VARS]

print(f"{'Threshold':>10}  {'N (Black)':>10}  {'Black coef':>11}  {'p-value':>8}  "
      f"{'Res x Black coef':>17}  {'p-value':>8}")
print(f"{BASELINE_PCT:>9}%  {int(df_restricted['mblack_1945def'].sum()):>10}  "
      f"{baseline_raw.params['Black']:>11.4f}  {baseline_raw.pvalues['Black']:>8.3f}  "
      f"{baseline_raw.params['Residential x Black']:>17.4f}  {baseline_raw.pvalues['Residential x Black']:>8.3f}  (baseline)")

threshold_results = {}
for pct in THRESHOLDS_PCT:
    if pct == BASELINE_PCT:
        threshold_results[f'{pct}\\%'] = baseline_results
        continue

    df_thresh = df_restricted.copy()
    df_thresh['mblack_1945def'] = np.where(df_thresh['pct_black'] >= pct / 100, 1, 0)
    df_thresh['ResidentialxBlack'] = df_thresh['Residential'] * df_thresh['mblack_1945def']

    results, raw, _ = fit_spec(df_thresh)
    threshold_results[f'{pct}\\%'] = results

    print(f"{pct:>9}%  {int(df_thresh['mblack_1945def'].sum()):>10}  "
          f"{raw.params['Black']:>11.4f}  {raw.pvalues['Black']:>8.3f}  "
          f"{raw.params['Residential x Black']:>17.4f}  {raw.pvalues['Residential x Black']:>8.3f}")

# order columns by threshold ascending, with baseline in its natural place
ordered = {f'{pct}\\%': threshold_results[f'{pct}\\%'] for pct in THRESHOLDS_PCT}

export_multiple_regressions(
    ordered,
    caption='Robustness to Majority-Black Threshold (\\% Black Required for mblack\\_1945def = 1)',
    label='tab:mblack_threshold_robustness',
    leaveout=leaveout_except(columns, keep=keep_labels),
)

print('\nsaved: tables/mblack_threshold_robustness.tex')
