import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + rent + valueh'

results_pct = smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3')
with open('tables/intial_results.md', 'w') as f:
    f.write(results_pct.summary().as_latex())