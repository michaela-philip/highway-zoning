import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

atl_sample = atl_sample.rename(columns={
    'mblack_mean_pct': 'Majority Black (pct)',
    'mblack_mean_share': 'Majority Black (share)'
})

model_pct = 'Highway ~ Majority Black + Residential + (Majority Black * Residential) + Median Rent + Median Home Value)'

results_pct = smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3').fit()
results_pct.summary().to_markdown('tables/intial_results.md', floatfmt=".2f", tablefmt="pipe")