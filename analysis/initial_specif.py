import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + rent + valueh + distance_to_cbd'
model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + rent + valueh + distance_to_cbd'

results_pct = smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3')
results_share = smf.ols(model_share, data=atl_sample).fit(cov_type='HC3')

results = pd.concat([results_pct.summary2(title = 'Mean Percentage').tables[1],
                     results_share.summary2(title = 'Mean Share').tables[1]])

results = results.T.rename(columns={
    'rent': 'Rent',
    'valueh': 'Home Value',
    'hwy': 'Highway',
    'mblack_mean_pct': 'Majority Black (Percent)',
    'mblack_mean_share':'Majority Black (Share)',
    'mblack_mean_pct:Residential': 'Majority Black (Percent) x Residential',
    'mblack_mean_share:Residential':'Majority Black (Share) x Residential',
    'distance_to_cbd' : 'Distance to CBD'
}).T

print(results)

results.to_latex('tables/initial_results.tex', float_format="%.3f",
                 column_format='lcccccc', 
                 caption = 'Initial Results',
                 label = 'tab:initial_results',
                 position = 'h',
                 escape=False)