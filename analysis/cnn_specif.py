import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results

df = pd.read_pickle('data/output/sample.pkl')
df['rent'] = df['rent'].replace(0, np.nan)
df['valueh'] = df['valueh'].replace(0, np.nan)
df = df.dropna(subset = ['rent', 'valueh']).copy()
from cnn.pick_counterfactuals import counterfactuals

controls = df.loc[df['grid_id'].isin(counterfactuals)].copy()
treated = df.loc[df['hwy']==1].copy()
sample = pd.concat([controls, treated], ignore_index=True)

model_1945def = 'hwy ~ mblack_1945def + Residential + (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + C(city)'
results_1945def = format_regression_results(smf.ols(model_1945def, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

# model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + C(city)'
# results_pct = format_regression_results(smf.ols(model_pct, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

# model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + C(city)'
# results_share = format_regression_results(smf.ols(model_share, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

# # other results together ?
# export_multiple_regressions([results_1945def, results_pct, results_share],
#                             caption = 'Determinants of Highway Placement',
#                             label = 'tab:cnn_results',
#                             leaveout = ['dist_water', 'owner'])

export_single_regression(results_1945def, caption = 'Determinants of Highway Placement', label = 'tab:cnn_pref_specif', widthmultiplier = 0.7, leaveout = ['dist_water', 'owner'])
