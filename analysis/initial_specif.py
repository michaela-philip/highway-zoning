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

# construct samples
from data_code.candidates import candidate_dict
out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (df['grid_id'].isin(candidates))].copy()
    treated = df.loc[(df['city'] == city) & (df['hwy']==1) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
    out_frames.append(treated)
sample = pd.concat(out_frames, ignore_index=True)

out_frames = []
for city in df['city'].unique():
    candidates = candidate_dict[city]
    controls = df.loc[(df['city'] == city) & (~df['grid_id'].isin(candidates))].copy()
    out_frames.append(controls)
inv_sample = pd.concat(out_frames, ignore_index=True)

# run and export regressions
model_1945def = 'hwy ~ mblack_1945def + Residential + (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_wholesample = format_regression_results(smf.ols(model_1945def, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['city']}))

model_1945def = 'hwy ~ (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_inv = format_regression_results(smf.ols(model_1945def, data=inv_sample).fit(cov_type='cluster', cov_kwds={'groups': inv_sample['city']}))

model_1945def = 'hwy ~ (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + dist_water + owner + C(city)'
results_effic = format_regression_results(smf.ols(model_1945def, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']}))

# wholesample regression in its own table
export_single_regression(results_wholesample, caption = 'Determinants of Highway Placement - Whole Sample', label = 'tab:wholesample_results', widthmultiplier=0.7, leaveout = ['dist_water', 'owner'])

# other results together
export_multiple_regressions({"`Efficient' Sample": results_effic, "`Inefficient' Sample": results_inv},
                            caption = "Determinants of Highway Placement - `Efficiency' Restricted Samples",
                            label = 'tab:effic_results',
                            leaveout = ['dist_water', 'owner'])