import pandas as pd
import numpy as np

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

atl_sample = atl_sample.rename(columns={
    'rent': 'Median Rent',
    'valueh': 'Median Home Value',
    'black_pop': 'Black Population',
    'share_black': 'Share of Black Residents',
    'numprec': 'Residents',
    'serial': 'Households',
    'pct_black': 'Percent Black',
    'hwy': 'Highway Present'
})
atl_sample = atl_sample.dropna(subset = 'Residents')

rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 
        'Percent Black', 'Highway Present', 'Residential']

sum_stats = pd.DataFrame({
    'Mean': atl_sample[rows].mean(),
    'Std': atl_sample[rows].std(),
    'Min': atl_sample[rows].min(),
    'Max': atl_sample[rows].max(),
    'N': atl_sample[rows].count(),
})

columns = ['Mean', 'Std', 'Min', 'Max', 'N']
sum_stats = sum_stats[columns]

print(sum_stats)
sum_stats.to_latex('tables/summary_stats.tex', float_format="%.3f",
                  column_format='lcccccc', 
                  escape=False, 
                  caption='Summary Statistics for Atlanta Sample',
                  label='tab:summary_stats')