import pandas as pd
import numpy as np
import re

# function to export df as latex table with full page width and add'l formatting
def export_latex_table(df, columns, caption, label):
    df = df[columns]
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2).to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = text.replace('\\begin{tabular}', '\\begin{tabular*}{\\linewidth}').replace('\\end{tabular}', '\\end{tabular*}')
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)

####################################################################################################

sample = pd.read_pickle('data/output/sample.pkl')

sample = sample.rename(columns={
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
sample = sample.dropna(subset = 'Residents')

rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 'Percent Black', 
        'Highway Present (1940)','Highway Present (1959)', 'Highway Constructed (1940-1959)', 'Residential']

sum_stats = pd.DataFrame({
    'Mean': sample[rows].mean(),
    'Std': sample[rows].std(),
    'N': sample[rows].count(),
})
columns = ['Mean', 'Std', 'N']

# sum_stats['Mean'] = sum_stats.apply(
#     lambda row: f"\\makecell[tr]{{{row['Mean']:.2f} \\\\ ({row['Std']:.2f})}}", axis=1)
export_latex_table(sum_stats, columns = columns, caption = 'Sample Grid Summary Statistics', label = 'tab:summary_stats')

# summary statistics by zoning designation
rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 
        'Percent Black', 'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)']

zoning_mean = sample.groupby('Residential')[rows].agg('mean')
zoning_std = sample.groupby('Residential')[rows].agg('std')
zoning_stats = pd.DataFrame({
    'Industrial': zoning_mean.T[0],
    'std_i':zoning_std.T[0],
    'Residential':zoning_mean.T[1],
    'std_r':zoning_std.T[1]
})
columns = ['Industrial', 'Residential']

zoning_stats['Industrial'] = zoning_stats.apply(
    lambda row: f"\\makecell[tr]{{{row['Industrial']:.2f} \\\\ ({row['std_i']:.2f})}}", axis=1)
zoning_stats['Residential'] = zoning_stats.apply(
    lambda row: f"\\makecell[tr]{{{row['Residential']:.2f} \\\\ ({row['std_r']:.2f})}}", axis=1)
export_latex_table(zoning_stats, columns = columns, caption = 'Summary Statistics by Zoning Designation', label = 'tab:summary_stats_zone')

# summary statistics by race designation
rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 
        'Residential', 'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)']

zoning_mean = sample.groupby('mblack_1945def')[rows].agg('mean')
zoning_std = sample.groupby('mblack_1945def')[rows].agg('std')
zoning_n = sample.groupby('mblack_1945def')['Households'].sum().T
print(zoning_n)
zoning_stats = pd.DataFrame({
    'White': zoning_mean.T[0],
    'std_w':zoning_std.T[0],
    'Black':zoning_mean.T[1],
    'std_b':zoning_std.T[1]
})
columns = ['White', 'Black']

zoning_stats.loc['Total Households'] = [
    zoning_n[0],  # Total for White
    None,         # Placeholder for std_w
    zoning_n[1],  # Total for Black
    None          # Placeholder for std_b
]
zoning_stats['White'] = zoning_stats.apply(
    lambda row: f"\\makecell[tr]{{{row['White']:.2f} \\\\ ({row['std_w']:.2f})}}", axis=1)
zoning_stats['Black'] = zoning_stats.apply(
    lambda row: f"\\makecell[tr]{{{row['Black']:.2f} \\\\ ({row['std_b']:.2f})}}", axis=1)
export_latex_table(zoning_stats, columns = columns, caption = 'Summary Statistics by Racial Designation', label = 'tab:summary_stats_race')