import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.lib.data import load_sample
from helpers.latex_formatting import export_table

RENAME = {
    'rent': 'Median Rent',
    'valueh': 'Median Home Value',
    'black_pop': 'Black Population',
    'share_black': 'Share of Black Residents',
    'numprec': 'Residents',
    'serial': 'Households',
    'pct_black': 'Percent Black',
    'hwy_40': 'Highway Present (1940)',
    'hwy_59': 'Highway Present (1959)',
    'hwy': 'Highway Constructed (1940-1959)',
}


### FUNCTION TO BUILD A MEAN(STD) SUMMARY TABLE SPLIT BY A BINARY GROUPING VARIABLE ###
### group_labels maps the group_var's 0/1 values to display column names, e.g.
### {0: 'Industrial', 1: 'Residential'} -- adds a Total Households row from the raw sum ###
def group_summary_table(df, group_var, group_labels, rows):
    means = df.groupby(group_var)[rows].mean()
    stds = df.groupby(group_var)[rows].std()
    totals = df.groupby(group_var)['Households'].sum()

    table = pd.DataFrame({
        col_label: [f"\\makecell[tr]{{{means.loc[val, row]:.2f}  ({stds.loc[val, row]:.2f})}}" for row in rows]
        for val, col_label in group_labels.items()
    }, index=rows)
    table.loc['Total Households'] = [f"{totals.loc[val]:.0f}" for val in group_labels]
    table.loc['Total Squares'] = [f"{len(df.loc[df[group_var] == val]):.0f}" for val in group_labels]
    return table


sample = load_sample().rename(columns=RENAME)
sample = sample.dropna(subset='Residents')

# --- whole-sample summary statistics ---
whole_sample_rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 'Percent Black',
                      'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)',
                      'Residential']
sum_stats = pd.DataFrame({
    'Mean': sample[whole_sample_rows].mean(),
    'Std': sample[whole_sample_rows].std(),
    'N': sample[whole_sample_rows].count(),
})
notes = "This table contains summary statistics for the full sample of grid squares. The table reports the mean, standard deviation, and number of observations for each variable."
export_table(sum_stats, caption='Sample Grid Summary Statistics', label='tab:summary_stats', notes = notes)

# --- summary statistics by zoning designation ---
zoning_rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 'Percent Black',
               'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)']
zoning_table = group_summary_table(sample, 'Residential', {0: 'Industrial', 1: 'Residential'}, zoning_rows)
notes = "This table contains summary statistics for the full sample of grid squares, split by zoning designation. The table reports the mean and standard deviation (in parentheses) for each variable, as well as the total number of households in each zoning category."
export_table(zoning_table, caption='Summary Statistics by Zoning Designation', label='tab:summary_stats_zone', notes = notes)

# --- summary statistics by racial designation ---
race_rows = ['Residents', 'Households', 'Median Rent', 'Median Home Value', 'Residential',
             'Highway Present (1940)', 'Highway Present (1959)', 'Highway Constructed (1940-1959)']
race_table = group_summary_table(sample, 'mblack_1945def', {0: 'White', 1: 'Black'}, race_rows)
notes = "This table contains summary statistics for the full sample of grid squares, split by racial designation. The table reports the mean and standard deviation (in parentheses) for each variable, as well as the total number of households in each racial category."
export_table(race_table, caption='Summary Statistics by Racial Designation', label='tab:summary_stats_race', notes = notes)

print('saved: tables/summary_stats.tex, tables/summary_stats_zone.tex, tables/summary_stats_race.tex')
