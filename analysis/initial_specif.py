import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

from helpers.latex_formatting import export_single_regression, export_multiple_regressions, format_regression_results

df = pd.read_pickle('data/output/sample.pkl')
from data_code.candidates import candidates_dict

out_frames = []
for city in df['city'].unique():
    candidates = candidates_dict[city]
    controls = df.loc[(df['city'] == city) & (~df['grid_id'].isin(candidates))].copy()
    treated = df.loc[(df['city'] == city) & (df['hwy']==1)].copy()
    out_frames.append(controls)
    out_frames.append(treated)
    
sample = pd.concat(out_frames, ignore_index=True)

model_naive = 'hwy ~ mblack_1945def + Residential + np.log(rent) + np.log(valueh) + distance_to_cbd + dist_to_hwy'
results_naive = format_regression_results(smf.ols(model_naive, data=sample).fit(cov_type='HC3'))

model_1945def = 'hwy ~ mblack_1945def + Residential + (mblack_1945def * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd + dist_to_hwy'
results_1945def = format_regression_results(smf.ols(model_1945def, data=sample).fit(cov_type='HC3'))

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd + dist_to_hwy'
results_pct = format_regression_results(smf.ols(model_pct, data=sample).fit(cov_type='HC3'))

model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + np.log(rent) + np.log(valueh) + distance_to_cbd + dist_to_hwy'
results_share = format_regression_results(smf.ols(model_share, data=sample).fit(cov_type='HC3'))

# naive regression in its own table
export_single_regression(results_naive, caption = 'Naive Regression Results', label = 'tab:naive_results', widthmultiplier=0.7)

# other results together ?
export_multiple_regressions([results_1945def, results_pct, results_share],
                            caption = 'Determinants of Highway Placement',
                            label = 'tab:initial_results')
