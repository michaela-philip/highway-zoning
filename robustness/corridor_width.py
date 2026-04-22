import itertools
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

# same as original candidate list creation with optional buffer to widen corridor
def create_candidate_list_buffered(data, cbd, buffer_width_m=0):
    pts = data.loc[data['hwy_40'] == 1].copy()
    if pts.empty:
        return []
    
    centroids = pts.geometry.centroid.reset_index(drop=True)
    n = len(centroids)
    lines = []
    
    # highway-to-highway connections
    for i, j in itertools.combinations(range(n), 2):
        p1 = centroids.iloc[i]
        p2 = centroids.iloc[j]
        lines.append(LineString([(p1.x, p1.y), (p2.x, p2.y)]))
    
    # highway-to-CBD connections
    cbd_point = cbd.geometry.iloc[0]
    for p in centroids:
        lines.append(LineString([(p.x, p.y), (cbd_point.x, cbd_point.y)]))
    
    # buffer lines if requested
    if buffer_width_m > 0:
        lines_buffered = [line.buffer(buffer_width_m) for line in lines]
    else:
        lines_buffered = lines
    
    rays = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries(lines_buffered, crs=data.crs)
    )
    
    # intersect with grid
    candidates = gpd.sjoin(
        data, rays, how='inner', predicate='intersects'
    ).drop_duplicates('grid_id')
    
    # elevation restriction
    elev_z = stats.zscore(candidates['dm_elevation'])
    candidates = candidates.loc[
        (elev_z > -1) & (elev_z < 1)
    ].copy()
    
    # drop 1940 highway squares
    candidates = candidates.loc[candidates['hwy_40'] == 0].copy()
    
    return candidates['grid_id'].unique().tolist()


def run_robustness_corridor_width(grid, centroids,
                                   demo_vars, geo_controls,
                                   buffer_widths_m,
                                   n_bootstrap=500, seed=42):
    np.random.seed(seed)
    results = []

    for bw in buffer_widths_m:
        print(f"\nBuffer width: {bw}m")
        
        # construct candidate list at this buffer width
        candidate_list_bw = {}
        for city in grid['city'].unique():
            city_data = grid[grid['city'] == city].copy()
            city_mask = (centroids['place'].str.lower()
                         .str.replace(' ', '') == 
                         city.lower().replace(' ', ''))
            city_cbd = centroids[city_mask]
            candidate_list_bw[city] = create_candidate_list_buffered(
                city_data, city_cbd, buffer_width_m=bw
            )
        
        # construct direct/indirect samples
        all_cand_ids = set( 
            gid for ids in candidate_list_bw.values() for gid in ids
        )
        
        df_direct   = grid[grid['grid_id'].isin(all_cand_ids)].copy()
        df_indirect = grid[~grid['grid_id'].isin(all_cand_ids)].copy()
        
        pct_direct = len(df_direct) / len(grid)
        n_hwy_direct   = df_direct['hwy'].sum()
        n_hwy_indirect = df_indirect['hwy'].sum()
        
        print(f"  Direct: {len(df_direct):,} squares ({pct_direct:.1%}), "
              f"{n_hwy_direct} hwy")
        print(f"  Indirect: {len(df_indirect):,} squares, "
              f"{n_hwy_indirect} hwy")
        
        if n_hwy_indirect < 5:
            print("  Skipping — too few highway squares in indirect sample")
            continue
        
        # bootstrap LPM on indirect sample
        city_dum = pd.get_dummies(
            df_indirect['city'], drop_first=True
        ).astype(float)
        
        X = pd.concat([
            pd.Series(1.0, index=df_indirect.index, name='const'),
            df_indirect[demo_vars + geo_controls],
            city_dum
        ], axis=1).astype(float)
        y = df_indirect['hwy'].astype(float)
        
        col_names = list(X.columns)
        boot_coefs = {v: [] for v in demo_vars}
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(df_indirect), 
                                    len(df_indirect), replace=True)
            X_b = X.iloc[idx].copy()
            y_b = y.iloc[idx].copy()
            X_b.index = range(len(X_b))
            y_b.index = range(len(y_b))
            X_b = X_b.loc[:, X_b.nunique() > 1]
            
            try:
                beta = np.linalg.lstsq(
                    X_b.values, y_b.values, rcond=None
                )[0]
                cn = list(X_b.columns)
                for v in demo_vars:
                    if v in cn:
                        boot_coefs[v].append(beta[cn.index(v)])
            except Exception:
                continue
        
        row = {
            'buffer_m'    : bw,
            'pct_direct'  : pct_direct,
            'n_direct'    : len(df_direct),
            'n_indirect'  : len(df_indirect),
            'n_hwy_direct'  : n_hwy_direct,
            'n_hwy_indirect': n_hwy_indirect,
        }
        for v in demo_vars:
            coefs = np.array(boot_coefs[v])
            row[f'{v}_coef'] = np.mean(coefs)
            row[f'{v}_se']   = np.std(coefs)
            row[f'{v}_p']    = 2 * min(
                (coefs > 0).mean(), (coefs < 0).mean()
            )
        
        results.append(row)
        
        key = 'ResidentialxBlack'
        print(f"  {key}: {row[f'{key}_coef']:+.4f} "
              f"(SE={row[f'{key}_se']:.4f}, "
              f"p={row[f'{key}_p']:.3f})")

    return pd.DataFrame(results)


# --- run across corridor widths ---
# buffer_width=0 replicates your original approach (line intersection)
# then widen progressively to see where the effect appears/disappears
grid = pd.read_pickle('data/output/sample.pkl')
centroids = pd.read_csv('data/input/msas_with_central_city_cbds.csv')
centroids = gpd.GeoDataFrame(centroids, geometry = gpd.points_from_xy(centroids.cbd_retail_long, centroids.cbd_retail_lat), 
                             crs = 'EPSG:4267') # best guess at CRS based off of projfinder.com
# prep variables as in LPM
grid['rent'] = grid['rent'].replace(0, 0.00001)
grid['valueh'] = grid['valueh'].replace(0, 0.00001)
grid['log_valueh'] = np.log(grid['valueh']) * grid['valueh_avail']
grid['log_rent'] = np.log(grid['rent']) * grid['rent_avail']
grid['city_louisville'] = (grid['city'] == 'louisville').astype(int)
grid['city_littlerock'] = (grid['city'] == 'littlerock').astype(int)
grid['distance_to_cbd_sq'] = grid['distance_to_cbd'] ** 2
grid['log_dist_to_rr'] = np.log(grid['dist_to_rr'])
grid['log_dist_to_rr_sq'] = grid['log_dist_to_rr'] ** 2
grid['log_dist_to_hwy'] = np.log(grid['dist_to_hwy'])
grid['ResidentialxBlack'] = grid['Residential'] * grid['mblack_1945def']

buffer_widths = [0, 150, 300, 500, 750, 1000, 1500, 2000]

robustness_df = run_robustness_corridor_width(
    grid       = grid,
    centroids  = centroids,
    demo_vars  = ['Residential', 'mblack_1945def', 
                  'ResidentialxBlack', 'log_valueh', 'log_rent'],
    geo_controls = ['log_dist_to_rr', 'log_dist_to_rr_sq', 'log_dist_to_hwy', 'distance_to_cbd', 
                    'distance_to_cbd_sq', 'flood_risk', 'dist_water', 'slope', 'dm_elevation', 'owner', 
                    'numprec'],
    buffer_widths_m = buffer_widths,
    n_bootstrap = 500
)

print("\n=== Robustness Summary ===")
print(robustness_df[[
    'buffer_m', 'pct_direct', 'n_hwy_indirect',
    'ResidentialxBlack_coef', 'ResidentialxBlack_se',
    'ResidentialxBlack_p'
]].round(4).to_string(index=False))

# --- plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# panel 1: coefficient and CI across buffer widths
ax = axes[0]
coefs = robustness_df['ResidentialxBlack_coef']
ses   = robustness_df['ResidentialxBlack_se']
xs    = robustness_df['buffer_m']

ax.fill_between(xs, coefs - 1.96*ses, coefs + 1.96*ses,
                alpha=0.2, color='tomato')
ax.plot(xs, coefs, 'o-', color='tomato', linewidth=2, markersize=7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)

# mark your current implicit width
# (the width that gives ~53% direct is your baseline)
ax.axvline(0, color='steelblue', linewidth=1.5, 
           linestyle=':', label='Current approach (line intersection)')
ax.set_xlabel('Corridor buffer width (meters)', fontsize=11)
ax.set_ylabel('Residential×Black coefficient\n(indirect sample)', fontsize=10)
ax.set_title('Robustness to Corridor Width\nResidential×Black in Indirect Sample',
             fontsize=11)
ax.legend(fontsize=9)

# panel 2: fraction of sample in direct path
ax2 = axes[1]
ax2.plot(xs, robustness_df['pct_direct'] * 100, 
         'o-', color='steelblue', linewidth=2, markersize=7)
ax2.axhline(53, color='steelblue', linewidth=1, 
            linestyle=':', label='Current (53%)')
ax2.set_xlabel('Corridor buffer width (meters)', fontsize=11)
ax2.set_ylabel('% of squares in direct path', fontsize=11)
ax2.set_title('Sample Composition by Corridor Width', fontsize=11)
ax2.legend(fontsize=9)

# annotate highway counts
for _, row in robustness_df.iterrows():
    ax2.annotate(f"n_hwy={int(row['n_hwy_indirect'])}",
                 (row['buffer_m'], row['pct_direct']*100),
                 textcoords='offset points', xytext=(0, 8),
                 fontsize=7, ha='center')

plt.tight_layout()
plt.savefig('tables/corridor_robustness.png', dpi=100, bbox_inches='tight')
plt.show()

print('\nsaved: tables/corridor_robustness.png')