import pandas as pd
import numpy as np

def export_scaled_latex(df, path, method="adjustbox", **kwargs):
    table = df.to_latex(**kwargs)
    if method == "adjustbox":
        wrapped = "\\begin{adjustbox}{width=\\linewidth}\n" + table + "\\end{adjustbox}\n"
    elif method == "resizebox":
        wrapped = "\\resizebox{\\linewidth}{!}{%\n" + table + "}\n"
    else:
        raise ValueError("Invalid method")
    with open(path, "w") as f:
        f.write(wrapped)

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
# format and save tex file
sum_stats.style.format(precision=2).to_latex(column_format='lcccccc', position_float = 'centering',
                caption='Sample Grid Summary Statistics', position = 'h', label='tab:summary_stats', hrules=True)

export_scaled_latex(sum_stats.style.format(precision=2), 'tables/summary_stats.tex', column_format='lcccccc', position_float = 'centering',
                caption='Sample Grid Summary Statistics', position = 'h', label='tab:summary_stats', hrules=True)

# summary statistics by zoning designation
rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 
        'Percent Black', 'Highway Present']

agg_funcs = {
    'Residents':'mean',
    'Households': 'mean',
    'Median Rent' : 'mean',
    'Median Home Value': 'mean',
    'Percent Black': 'mean',
    'Highway Present':'mean'
}

zoning_sum = atl_sample.groupby('Residential')[rows].agg(agg_funcs).T
zoning_stats = pd.DataFrame({
    'Industrial': zoning_sum[0],
    'Residential':zoning_sum[1]
})
columns = ['Industrial', 'Residential']
zoning_stats = zoning_stats[columns]
print(zoning_stats)

zoning_stats.style.format(precision=2).to_latex('tables/summary_stats_zone.tex', 
                column_format='lcc', 
                position_float = 'centering',
                caption='Summary Statistics by Zoning Designation',
                position = 'h',
                label='tab:summ_stats_zone',
                hrules=True)