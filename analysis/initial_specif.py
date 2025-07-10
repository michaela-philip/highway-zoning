import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

model_pct = 'hwy ~ mblack_mean_pct + Residential + (mblack_mean_pct * Residential) + rent + valueh'
model_share = 'hwy ~ mblack_mean_share + Residential + (mblack_mean_share * Residential) + rent + valueh'
model_med_pct = 'hwy ~ mblack_median_pct + Residential + (mblack_median_pct * Residential) + rent + valueh'
model_med_share = 'hwy ~ mblack_median_share + Residential + (mblack_median_share * Residential) + rent + valueh'

results_pct = smf.ols(model_pct, data=atl_sample).fit(cov_type='HC3')
results_share = smf.ols(model_share, data=atl_sample).fit(cov_type='HC3')
results_med_pct = smf.ols(model_med_pct, data=atl_sample).fit(cov_type='HC3')
results_med_share = smf.ols(model_med_share, data=atl_sample).fit(cov_type='HC3')
results = pd.concat([results_pct.summary2(title = 'Mean Percentage').tables[1],
                     results_share.summary2(title = 'Mean Share').tables[1],
                     results_med_pct.summary2(title = 'Median Percentage').tables[1],
                     results_med_share.summary2(title = 'Mean Percentage').tables[1]])

results = results.T.rename(columns={
    'rent': 'Rent',
    'valueh': 'Home Value',
    'hwy': 'Highway',
    'mblack_mean_pct': 'Black (Mean Pct)',
    'mblack_mean_share':'Black (Mean Share)',
    'mblack_median_pct' : 'Black (Median Pct)',
    'mblack_median_share' : 'Black (Median Share)',
    'mblack_mean_pct:Residential': 'Black (Mean Pct) x Residential',
    'mblack_mean_share:Residential':'Black (Mean Share) x Residential',
    'mblack_median_pct:Residential' : 'Black (Median Pct) x Residential',
    'mblack_median_share:Residential' : 'Black (Median Share) x Residential'
}).T

print(results)

results.to_latex('tables/initial_results.tex', float_format="%.3f",
                 column_format='lcccccc', 
                 escape=False)