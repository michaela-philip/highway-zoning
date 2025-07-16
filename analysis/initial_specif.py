import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

model_50_pct = 'hwy ~ mblack_50_pct + Residential + (mblack_50_pct * Residential) + rent + valueh + distance_to_cbd'
model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + rent + valueh + distance_to_cbd'
model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + rent + valueh + distance_to_cbd'

results_50_pct = smf.ols(model_50_pct, data=atl_sample).fit(cov_type='HC3')
results_pct = smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3')
results_share = smf.ols(model_share, data=atl_sample).fit(cov_type='HC3')

# initial results with >= 50% black population threshold
results = results_50_pct.summary2(title = 'Mean 50% Threshold').tables[1]
results = results.T.rename(columns={
    'Rent': 'rent',
    'Home Value': 'valueh',
    'Highway': 'hwy',
    'mblack_50_pct': 'Majority Black (50% Threshold)',
    'mblack_50_pct:Residential': 'Majority Black (50% Threshold) x Residential',
    'distance_to_cbd':'Distance to CBD' 
}).T
print(results)

results.style.format(precision=3).to_latex('tables/initial_results.tex',
                 column_format='lcccccc', 
                 caption = 'Initial Results',
                 label = 'tab:initial_results',
                 position = 'h',
                 position_float = 'centering',
                 hrules = True)

# second results with alt. definitions of majority Black
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

results.style.format(precision=3).to_latex('tables/second_results.tex',
                 column_format='lcccccc', 
                 caption = 'Alternate Definitions of Majority Black',
                 label = 'tab:second_results',
                 position = 'h',
                 position_float = 'centering',
                 hrules = True)