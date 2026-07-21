import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.latex_formatting import export_single_regression, format_regression_results
from analysis.lib.data import load_sample, restrict_to_discretionary
from analysis.lib.specs import (
    CORE_VARS, HOUSING_VARS, GEO_CONTROLS, HH_CONTROLS, build_spec, leaveout_except, fit_ols,
)

df = load_sample()
df_restricted = restrict_to_discretionary(df)

x_vars, columns = build_spec(df_restricted, CORE_VARS, HOUSING_VARS, GEO_CONTROLS, HH_CONTROLS)

results_wholesample = format_regression_results(fit_ols(df_restricted, x_vars, columns))
export_single_regression(
    results_wholesample,
    caption='Determinants of Highway Placement - Full Sample',
    label='tab:wholesample_results',
    widthmultiplier=0.7,
    leaveout=leaveout_except(columns, keep=[label for _, label in CORE_VARS]),
)
