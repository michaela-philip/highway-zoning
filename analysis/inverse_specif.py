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

from data_code.candidates import candidate_dict
out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (~df['grid_id'].isin(candidates))].copy()
    treated = df.loc[(df['city'] == city) & (df['hwy']==1) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
    out_frames.append(treated)
sample = pd.concat(out_frames, ignore_index=True)

model_1945def = 'hwy ~ (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_1945def = format_regression_results(smf.ols(model_1945def, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_pct = format_regression_results(smf.ols(model_pct, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_share = format_regression_results(smf.ols(model_share, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

# other results together ?
# export_multiple_regressions([results_1945def, results_pct, results_share],
#                             caption = 'Determinants of Highway Placement',
#                             label = 'tab:inverse_results',
#                             leaveout = ['dist_water', 'owner'])

export_single_regression(results_1945def, caption = 'Determinants of Highway Placement', label = 'tab:inverse_specif', widthmultiplier = 0.7, leaveout = ['dist_water', 'owner'])