import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.latex_formatting import export_single_regression
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.bootstrap import fit_ols
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS,
    build_spec, leaveout_except,
)

df = load_sample()

x_vars, columns = build_spec(df, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, LOG_DIST_HWY, HH_CONTROLS)

results_wholesample, *_ = fit_ols(df, x_vars, columns)
notes = "This table contains estimates of the impact of residential zoning and majority-Black status on the likelihood of highway placement using a linear probability model on the full sample of grid squares. " \
"Standard errors are reported in parenthesis and estimated using OLS. The model includes controls for housing, geographic, and demographic characteristics, as well as city fixed effects." \

keep = [label for _, label in CORE_VARS + HOUSING_VARS + GEO_CONTROLS + LOG_DIST_HWY + HH_CONTROLS]

export_single_regression(
    results_wholesample,
    caption='Determinants of Highway Placement - Control Variables Only',
    label='tab:wholesample_results_no_cnn',
    widthmultiplier=0.7,
    leaveout=leaveout_except(columns, keep=[label for _, label in CORE_VARS]),
    notes = notes
)