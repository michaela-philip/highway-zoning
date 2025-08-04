import pandas as pd
import numpy as np
import re

# function to export df as latex table with full page width and add'l formatting
def export_latex_table(df, caption, label):
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2).to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = text.replace('\\begin{tabular}', '\\begin{tabular*}{\\linewidth}').replace('\\end{tabular}', '\\end{tabular*}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)

####################################################################################################

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

atl_sample = atl_sample.rename(columns={
    'rent': 'Median Rent',
    'valueh': 'Median Home Value',
    'black_pop': 'Black Population',
    'share_black': 'Share of Black Residents',
    'numprec': 'Residents',
    'serial': 'Households',
    'pct_black': 'Percent Black',
    'hwy_40': 'Highway Present (1940)',
    'hwy_59':'Highway Present (1959)',
    'hwy':'Highway Constructed (1940-1959)'})
atl_sample = atl_sample.dropna(subset = 'Residents')

rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 'Percent Black', 
        'Highway Present (1940)','Highway Present (1959)', 'Highway Constructed (1940-1959)', 'Residential']

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
export_latex_table(sum_stats, caption = 'Sample Grid Summary Statistics', label = 'tab:summary_stats')

# summary statistics by zoning designation
rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 
        'Percent Black', 'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)']

# agg_funcs = {
#     'Residents':'mean',
#     'Households': 'mean',
#     'Median Rent' : 'mean',
#     'Median Home Value': 'mean',
#     'Percent Black': 'mean',
#     'Highway Present':'mean'
# }

zoning_sum = atl_sample.groupby('Residential')[rows].agg('mean').T
zoning_stats = pd.DataFrame({
    'Industrial': zoning_sum[0],
    'Residential':zoning_sum[1]
})
columns = ['Industrial', 'Residential']
zoning_stats = zoning_stats[columns]

print(zoning_stats)
export_latex_table(zoning_stats, caption = 'Summary Statistics by Zoning Designation', label = 'tab:summary_stats_zone')