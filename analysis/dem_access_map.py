import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.lib.data import load_sample, compute_characteristic_access, compute_shares
from analysis.lib.figures import plot_access_map, export_figure_tex

CITY = 'atlanta'

# dem_access built the same way as in exposure_blackshare.py, but over the whole sample
# (not just the indirect sample), then standardized within each city
cell_width = 150
df = load_sample(cell_width, impute = False)
city_grid = df[df['city'] == CITY]  # every square in the city, for the gray background
df = df[df['zoning_change'] != 1]

df = compute_shares(df)
df = compute_characteristic_access(df, 'share_black', 'dem_access', decay_m = 600, max_dist_m = 3000)
for city in df['city'].unique():
    city_mean = df.loc[df['city'] == city, 'dem_access'].mean()
    city_std = df.loc[df['city'] == city, 'dem_access'].std()
    df.loc[df['city'] == city, 'dem_access'] = (df.loc[df['city'] == city, 'dem_access'] - city_mean)/ city_std

plot_access_map(df[df['city'] == CITY], city_grid, 'dem_access',
                path = f'figures/dem_access_{CITY}.pdf',
                colorbar_label = 'Demographic Access to the Black Population (standardized)',
                excluded_label = 'Dropped (zoning change)')

notes = "Each square is a 150m grid cell in Atlanta, shaded by its distance-decayed access to the city's Black population " \
"(600m decay, 3km cutoff), standardized to mean zero and unit standard deviation within the city. The color scale is truncated " \
"at the 1st and 99th percentiles. Squares containing a highway built before 1940 are shown in dark gray; squares dropped from " \
"the sample because their zoning changed are shown in light gray."
export_figure_tex(f'figures/dem_access_{CITY}.pdf', caption = 'Demographic Access to the Black Population, Atlanta',
                  label = f'fig:dem_access_{CITY}', notes = notes)
