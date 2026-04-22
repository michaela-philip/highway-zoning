import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy import stats
import numpy as np
import pandas as pd
import glob
import os

grid = pd.read_pickle('data/output/sample.pkl')
dataroot = 'cnn/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv_files.sort(key=os.path.getmtime, reverse=True)
last_csv = csv_files[0]
filename_out = os.path.basename(last_csv)
print(f'Using most recent predictions from: {filename_out}')

def plot_propensity_overlap(results_csv_path, grid, temperature_values=[1.0, 5.0, 10.0, 20.0]):
    # load predictions
    preds = pd.read_csv(results_csv_path)
    
    # merge with true highway labels
    hwy_labels = grid[['grid_id', 'hwy']].drop_duplicates('grid_id')
    preds = preds.merge(hwy_labels, on='grid_id', how='left')
    preds['hwy'] = preds['hwy'].fillna(0).astype(int)

    probs_hwy1 = preds.loc[preds['hwy'] == 1, 'prob_hwy'].values
    probs_hwy0 = preds.loc[preds['hwy'] == 0, 'prob_hwy'].values

    print(f'Total cells: {len(preds)} | Highway: {len(probs_hwy1)} ({100*len(probs_hwy1)/len(preds):.1f}%) | Non-highway: {len(probs_hwy0)}')
    print(f'\nRaw probability summary:')
    print(f'  highway=1 : mean={probs_hwy1.mean():.4f}, p10={np.percentile(probs_hwy1,10):.4f}, p50={np.percentile(probs_hwy1,50):.4f}, p90={np.percentile(probs_hwy1,90):.4f}')
    print(f'  highway=0 : mean={probs_hwy0.mean():.4f}, p10={np.percentile(probs_hwy0,10):.4f}, p50={np.percentile(probs_hwy0,50):.4f}, p90={np.percentile(probs_hwy0,90):.4f}')

    # --- Plot 1: raw probabilities, linear and log scale ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Propensity Score Overlap — Raw Probabilities', fontsize=12)

    for ax, yscale in zip(axes, ['linear', 'log']):
        ax.hist(probs_hwy0, bins=100, alpha=0.5, density=True, label='highway=0', color='steelblue')
        ax.hist(probs_hwy1, bins=100, alpha=0.5, density=True, label='highway=1', color='tomato')
        ax.set_xlabel('predicted probability')
        ax.set_ylabel('density')
        ax.set_yscale(yscale)
        ax.legend()
        ax.set_title(f'scale={yscale}')

    plt.tight_layout()
    plt.savefig('cnn/figures/overlap_raw.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('saved: cnn/figures/overlap_raw.png')

# run it — point to your most recent prediction csv
# plot_propensity_overlap(
#     results_csv_path=dataroot + filename_out,  # uses the filename from your prediction script
#     grid=grid,
#     temperature_values=[1.0, 5.0, 10.0, 20.0]
# )

def check_restricted_sample(results_csv_path, grid, T=1.0, tau=0.10):
    preds = pd.read_csv(results_csv_path)
    hwy_labels = grid[['grid_id', 'hwy']].drop_duplicates('grid_id')
    preds = preds.merge(hwy_labels, on='grid_id', how='left')
    preds['hwy'] = preds['hwy'].fillna(0).astype(int)

    eps = 1e-6
    raw_probs = preds['prob_hwy'].values.clip(eps, 1 - eps)
    logits = np.log(raw_probs / (1 - raw_probs))
    scaled_probs = 1 / (1 + np.exp(-logits / T))
    
    restricted = preds[scaled_probs >= tau]
    
    n_total     = len(restricted)
    n_hwy1      = (restricted['hwy'] == 1).sum()
    n_hwy0      = (restricted['hwy'] == 0).sum()
    
    print(f'Restricted sample (T={T}, τ={tau}):')
    print(f'  Total squares : {n_total:,}')
    print(f'  highway = 1   : {n_hwy1:,} ({100*n_hwy1/n_total:.1f}%)')
    print(f'  highway = 0   : {n_hwy0:,} ({100*n_hwy0/n_total:.1f}%)')
    print(f'  Outcome mean  : {restricted["hwy"].mean():.4f}')

# check_restricted_sample(dataroot + filename_out, grid, T=5.0, tau=0.2)

def plot_geographic_similarity_conditional(preds_csv, grid, 
                                            geo_features=None,
                                            prob_bins=None,
                                            figsize=(16, 20)):
    if geo_features is None:
        geo_features = ['elevation', 'dist_water', 'dist_to_hwy', 'distance_to_cbd']
    
    if prob_bins is None:
        prob_bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]

    # --- load and merge ---
    preds = pd.read_csv(preds_csv)
    preds['grid_id'] = preds['grid_id'].astype(str)
    grid_copy = grid.copy()
    grid_copy['grid_id'] = grid_copy['grid_id'].astype(str)

    df = preds.merge(
        grid_copy[['grid_id', 'hwy'] + geo_features],
        on='grid_id', how='left'
    )
    df['hwy'] = df['hwy'].fillna(0).astype(int)
    df['prob_bin'] = pd.cut(df['prob_hwy'], bins=prob_bins, 
                             include_lowest=True)
    bin_labels = [str(b) for b in df['prob_bin'].cat.categories]
    n_bins     = len(bin_labels)
    n_features = len(geo_features)

    # --- layout ---
    # one row per feature, one column per bin
    # plus a summary row at the bottom (standardized mean difference)
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(n_features + 1, n_bins, 
                             hspace=0.5, wspace=0.35)

    feature_labels = {
        'elevation'      : 'Elevation (std)',
        'dist_water'     : 'Dist. to Water (std)',
        'dist_to_hwy'    : 'Dist. to Hwy (std)',
        'distance_to_cbd': 'Dist. to CBD (std)',
        'dist_to_rr'    : 'Dist. to RR (std)',
        'flood_risk'    : 'Flood Risk (std)'
    }

    smd_matrix = np.full((n_features, n_bins), np.nan)

    for col_idx, (bin_label, bin_cat) in enumerate(
            zip(bin_labels, df['prob_bin'].cat.categories)):

        bin_data = df[df['prob_bin'] == bin_cat]
        hwy1 = bin_data[bin_data['hwy'] == 1]
        hwy0 = bin_data[bin_data['hwy'] == 0]
        n1, n0 = len(hwy1), len(hwy0)

        for row_idx, feat in enumerate(geo_features):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            vals1 = hwy1[feat].dropna().values
            vals0 = hwy0[feat].dropna().values

            if len(vals1) > 1 and len(vals0) > 1:
                # check for zero variance — KDE fails on constant data
                if vals1.std() == 0:
                    # all highway squares in this bin have same value — plot as vline
                    ax.axvline(vals1[0], color='tomato', lw=2, label='hwy=1')
                    if vals0.std() > 0:
                        kde0 = stats.gaussian_kde(vals0)
                        x_range = np.linspace(vals0.min()-0.5, vals0.max()+0.5, 200)
                        ax.fill_between(x_range, kde0(x_range), alpha=0.4, color='steelblue')
                    smd = ((vals1.mean() - vals0.mean()) / 
                        (vals0.std() if vals0.std() > 0 else 1e-6))
                    smd_matrix[row_idx, col_idx] = smd
                    ax.set_title(f'SMD={smd:.2f}\n(const hwy)', fontsize=7, pad=2)
                elif vals0.std() == 0:
                    # all non-highway squares in this bin have same value
                    ax.axvline(vals0[0], color='steelblue', lw=2, label='hwy=0')
                    if vals1.std() > 0:
                        kde1 = stats.gaussian_kde(vals1)
                        x_range = np.linspace(vals1.min()-0.5, vals1.max()+0.5, 200)
                        ax.fill_between(x_range, kde1(x_range), alpha=0.4, color='tomato')
                    smd = ((vals1.mean() - vals0.mean()) / 
                        (vals1.std() if vals1.std() > 0 else 1e-6))
                    smd_matrix[row_idx, col_idx] = smd
                    ax.set_title(f'SMD={smd:.2f}\n(const non-hwy)', fontsize=7, pad=2)
                else:
                    # normal case — both have variation, proceed with KDE
                    x_min = min(vals1.min(), vals0.min())
                    x_max = max(vals1.max(), vals0.max())
                    x_range = np.linspace(x_min - 0.5, x_max + 0.5, 200)

                    kde1 = stats.gaussian_kde(vals1)
                    kde0 = stats.gaussian_kde(vals0)

                    ax.fill_between(x_range, kde1(x_range), 
                                    alpha=0.4, color='tomato', label='hwy=1')
                    ax.fill_between(x_range, kde0(x_range), 
                                    alpha=0.4, color='steelblue', label='hwy=0')
                    ax.plot(x_range, kde1(x_range), color='tomato', lw=1.2)
                    ax.plot(x_range, kde0(x_range), color='steelblue', lw=1.2)

                    ax.axvline(vals1.mean(), color='tomato', linestyle='--', lw=1, alpha=0.8)
                    ax.axvline(vals0.mean(), color='steelblue', linestyle='--', lw=1, alpha=0.8)

                    pooled_std = np.sqrt((vals1.std()**2 + vals0.std()**2) / 2)
                    smd = (vals1.mean() - vals0.mean()) / pooled_std if pooled_std > 0 else 0
                    smd_matrix[row_idx, col_idx] = smd
                    ax.set_title(f'SMD={smd:.2f}', fontsize=7, pad=2)

            # labels
            if col_idx == 0:
                ax.set_ylabel(feature_labels.get(feat, feat), fontsize=8)
            if row_idx == 0:
                ax.set_xlabel('')
                bin_str = str(bin_cat).replace('(','').replace(']','')
                lo, hi  = bin_str.split(',')
                ax.set_title(
                    f'p ∈ ({float(lo.strip()):.2f}, {float(hi.strip()):.2f}]\n'
                    f'n₁={n1}, n₀={n0}\nSMD={smd_matrix[row_idx,col_idx]:.2f}',
                    fontsize=7, pad=4
                )

            ax.tick_params(labelsize=6)
            ax.set_yticks([])

            if row_idx == 0 and col_idx == n_bins - 1:
                ax.legend(fontsize=6, loc='upper right')

    # --- summary row: SMD heatmap across bins and features ---
    ax_heat = fig.add_subplot(gs[n_features, :])
    im = ax_heat.imshow(
        np.abs(smd_matrix), aspect='auto',
        cmap='RdYlGn_r', vmin=0, vmax=0.5
    )
    ax_heat.set_xticks(range(n_bins))
    ax_heat.set_xticklabels(bin_labels, fontsize=8, rotation=30, ha='right')
    ax_heat.set_yticks(range(n_features))
    ax_heat.set_yticklabels(
        [feature_labels.get(f, f) for f in geo_features], fontsize=8
    )
    ax_heat.set_title(
        '|Standardized Mean Difference| by Probability Bin and Feature\n'
        '(green = similar, red = different — target: green everywhere)',
        fontsize=9
    )
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label='|SMD|')

    # annotate cells with values
    for i in range(n_features):
        for j in range(n_bins):
            if not np.isnan(smd_matrix[i, j]):
                ax_heat.text(j, i, f'{smd_matrix[i,j]:.2f}',
                             ha='center', va='center',
                             fontsize=7,
                             color='white' if abs(smd_matrix[i,j]) > 0.35 
                             else 'black')

    fig.suptitle(
        'Geographic Similarity of Highway and Non-Highway Squares\n'
        'Conditional on CNN Predicted Probability\n'
        '(distributions should overlap within each bin if CNN captures geography)',
        fontsize=11, y=1.01
    )

    plt.savefig('cnn/figures/geographic_similarity_conditional.png',
                dpi=100, bbox_inches='tight')
    plt.show()
    print('saved: cnn/figures/geographic_similarity_conditional.png')

    # --- print summary table ---
    print('\nStandardized Mean Difference Summary (hwy=1 vs hwy=0 within bin):')
    print('Target: |SMD| < 0.10 in all cells\n')
    smd_df = pd.DataFrame(
        smd_matrix,
        index=[feature_labels.get(f, f) for f in geo_features],
        columns=bin_labels
    )
    print(smd_df.round(3).to_string())
    print(f'\nMean |SMD| across all bins and features: '
          f'{np.nanmean(np.abs(smd_matrix)):.3f}')
    print(f'Max  |SMD|: {np.nanmax(np.abs(smd_matrix)):.3f}')

    return smd_df, smd_matrix


# # --- run it ---
smd_df, smd_matrix = plot_geographic_similarity_conditional(
    dataroot + filename_out,
    grid,
    geo_features = ['elevation', 'dist_water', 'dist_to_hwy', 'distance_to_cbd', 'dist_to_rr', 'flood_risk'],
    prob_bins  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
)

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

model1 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model1*.csv')), key=os.path.getmtime, reverse=True)[0]
model2 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model2*.csv')), key=os.path.getmtime, reverse=True)[0]
model3 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model3*.csv')), key=os.path.getmtime, reverse=True)[0]
model4 = sorted(glob.glob(os.path.join(dataroot, 'predicted_activation-model4*.csv')), key=os.path.getmtime, reverse=True)[0]


def plot_probability_heatmap(preds_csv, grid, 
                              cities=None,
                              prob_col='prob_hwy',
                              figsize_per_city=(10, 8),
                              cmap='YlOrRd',
                              threshold=0.05,
                              save_path='cnn/figures/probability_heatmap.png'):
    # --- load and merge ---
    preds = pd.read_csv(preds_csv)
    preds['grid_id'] = preds['grid_id'].astype(str)

    grid_plot = grid.copy()
    grid_plot['grid_id'] = grid_plot['grid_id'].astype(str)
    grid_plot = grid_plot.merge(
        preds[['grid_id', prob_col]], on='grid_id', how='left'
    )
    grid_plot[prob_col] = grid_plot[prob_col].fillna(0)

    if cities is None:
        cities = sorted(grid_plot['city'].unique())
    n_cities = len(cities)

    fig, axes = plt.subplots(
        n_cities, 1,
        figsize=(figsize_per_city[0], figsize_per_city[1] * n_cities),
        squeeze=False
    )

    # shared colormap and norm across all cities
    vmin = 0.0
    vmax = grid_plot[prob_col].quantile(0.99)  # clip top 1% for contrast

    norm = mcolors.PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax)

    # custom colormap: white -> yellow -> orange -> red -> dark red
    colors_list = ['#f7f7f7', '#fee08b', '#fc8d59', '#d73027', '#67001f']
    cmap_custom = LinearSegmentedColormap.from_list(
        'hwy_prob', colors_list, N=256
    )

    for row_idx, city in enumerate(cities):
        ax = axes[row_idx, 0]
        city_grid = grid_plot[grid_plot['city'] == city].copy()

        # reproject to a sensible local CRS for plotting if needed
        if city_grid.crs and city_grid.crs.is_geographic:
            city_grid = city_grid.to_crs(epsg=3857)

        # --- layer 1: all squares colored by probability ---
        city_grid.plot(
            column=prob_col,
            ax=ax,
            cmap=cmap_custom,
            norm=norm,
            linewidth=0,
            alpha=0.85
        )

        # --- layer 2: outline candidate pool (prob >= threshold) ---
        candidates = city_grid[
            (city_grid[prob_col] >= threshold) & 
            (city_grid['hwy'] == 0)
        ]
        if len(candidates) > 0:
            candidates.plot(
                ax=ax,
                facecolor='none',
                edgecolor='steelblue',
                linewidth=0.8,
                alpha=0.6
            )

        # --- layer 3: highway squares prominently marked ---
        hwy_squares = city_grid[city_grid['hwy'] == 1]
        if len(hwy_squares) > 0:
            hwy_squares.plot(
                ax=ax,
                facecolor='cyan',
                edgecolor='black',
                linewidth=0.8,
                alpha=0.95,
                zorder=5
            )

        ax.set_title(
            f'{city.title()} — CNN Geographic Suitability Probability \n',
            fontsize=10
        )
        ax.set_axis_off()

        # --- colorbar ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('P(highway | geography)', fontsize=9)

        # annotate with summary stats
        n_cand = len(candidates)
        n_hwy  = len(hwy_squares)
        mean_p_hwy  = city_grid.loc[city_grid['hwy']==1, prob_col].mean()
        mean_p_cand = candidates[prob_col].mean() if len(candidates) > 0 \
                      else float('nan')

        textstr = (f'Highway squares: {n_hwy}\n'
                   f'Candidate pool: {n_cand:,}\n'
                   f'Mean P (hwy squares): {mean_p_hwy:.3f}\n'
                   f'Mean P (candidates): {mean_p_cand:.3f}')
        ax.text(0.02, 0.02, textstr,
                transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.8, edgecolor='gray'))

    plt.suptitle(
        'Geographic Suitability for Highway Placement\n',
        fontsize=13, y=1.01, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {save_path}')

    return grid_plot


def plot_probability_heatmap_detailed(preds_csv, grid,
                                       city,
                                       prob_col='prob_hwy',
                                       threshold=0.05,
                                       n_quantile_panels=4,
                                       save_path=None):
    preds = pd.read_csv(preds_csv)
    preds['grid_id'] = preds['grid_id'].astype(str)

    grid_plot = grid[grid['city'] == city].copy()
    grid_plot['grid_id'] = grid_plot['grid_id'].astype(str)
    grid_plot = grid_plot.merge(
        preds[['grid_id', prob_col]], on='grid_id', how='left'
    )
    grid_plot[prob_col] = grid_plot[prob_col].fillna(0)

    if grid_plot.crs and grid_plot.crs.is_geographic:
        grid_plot = grid_plot.to_crs(epsg=3857)

    # quantile-based probability bins for discrete coloring
    quantile_labels = [f'Q{i+1}' for i in range(n_quantile_panels)]
    grid_plot['prob_bin'] = pd.qcut(
        grid_plot[prob_col],
        q=n_quantile_panels,
        labels=quantile_labels,
        duplicates='drop'
    )

    fig = plt.figure(figsize=(18, 12))

    # --- main map: continuous probability ---
    ax_main = fig.add_axes([0.0, 0.35, 0.55, 0.60])

    colors_list = ['#f7f7f7', '#fee08b', '#fc8d59', '#d73027', '#67001f']
    cmap_custom = LinearSegmentedColormap.from_list(
        'hwy_prob', colors_list, N=256
    )
    vmax = grid_plot[prob_col].quantile(0.99)
    norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=vmax)

    grid_plot.plot(
        column=prob_col, ax=ax_main,
        cmap=cmap_custom, norm=norm,
        linewidth=0, alpha=0.9
    )

    # candidate pool outline
    candidates = grid_plot[
        (grid_plot[prob_col] >= threshold) & (grid_plot['hwy'] == 0)
    ]
    if len(candidates) > 0:
        candidates.plot(ax=ax_main, facecolor='none',
                        edgecolor='dodgerblue', linewidth=0.5, alpha=0.7)

    # highway squares
    hwy_sq = grid_plot[grid_plot['hwy'] == 1]
    hwy_sq.plot(ax=ax_main, facecolor='cyan', edgecolor='black',
                linewidth=0.8, alpha=0.95, zorder=5)

    ax_main.set_title(f'{city.title()} — Continuous Probability',
                       fontsize=11)
    ax_main.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_main, shrink=0.6,
                       label='P(highway | geography)')

    # --- zoom panel: highest probability area ---
    ax_zoom = fig.add_axes([0.56, 0.35, 0.43, 0.60])

    # find bounding box of top-decile probability squares
    top_decile = grid_plot[
        grid_plot[prob_col] >= grid_plot[prob_col].quantile(0.90)
    ]
    if len(top_decile) > 0:
        minx, miny, maxx, maxy = top_decile.total_bounds
        pad_x = (maxx - minx) * 0.15
        pad_y = (maxy - miny) * 0.15

        grid_plot.plot(
            column=prob_col, ax=ax_zoom,
            cmap=cmap_custom, norm=norm,
            linewidth=0, alpha=0.9
        )
        if len(candidates) > 0:
            candidates.plot(ax=ax_zoom, facecolor='none',
                            edgecolor='dodgerblue',
                            linewidth=0.8, alpha=0.8)
        hwy_sq.plot(ax=ax_zoom, facecolor='cyan', edgecolor='black',
                    linewidth=1.0, alpha=0.95, zorder=5)

        ax_zoom.set_xlim(minx - pad_x, maxx + pad_x)
        ax_zoom.set_ylim(miny - pad_y, maxy + pad_y)
        ax_zoom.set_title('Zoom: Top 10% Probability Area', fontsize=11)
        ax_zoom.set_axis_off()

    # --- bottom: probability histogram by highway status ---
    ax_hist = fig.add_axes([0.0, 0.05, 0.55, 0.25])

    probs_hwy1 = grid_plot.loc[grid_plot['hwy'] == 1, prob_col]
    probs_hwy0 = grid_plot.loc[grid_plot['hwy'] == 0, prob_col]

    bins = np.linspace(0, grid_plot[prob_col].quantile(0.995), 50)
    ax_hist.hist(probs_hwy0, bins=bins, density=True,
                 alpha=0.5, color='steelblue', label='hwy=0')
    ax_hist.hist(probs_hwy1, bins=bins, density=True,
                 alpha=0.6, color='tomato', label='hwy=1')
    ax_hist.axvline(threshold, color='black', linestyle='--',
                    linewidth=1.2, label=f'threshold={threshold}')
    ax_hist.set_xlabel('Predicted Probability')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Probability Distribution by Highway Status')
    ax_hist.legend(fontsize=9)

    # --- bottom right: spatial autocorrelation of predictions ---
    ax_scatter = fig.add_axes([0.60, 0.05, 0.38, 0.25])

    # probability rank vs. distance to nearest highway square
    if len(hwy_sq) > 0:
        from scipy.spatial import cKDTree

        hwy_centroids = np.column_stack([
            hwy_sq.geometry.centroid.x.values,
            hwy_sq.geometry.centroid.y.values
        ])
        all_centroids = np.column_stack([
            grid_plot.geometry.centroid.x.values,
            grid_plot.geometry.centroid.y.values
        ])
        tree = cKDTree(hwy_centroids)
        dists, _ = tree.query(all_centroids, k=1)
        grid_plot['dist_to_nearest_hwy'] = dists

        sample = grid_plot[grid_plot['hwy'] == 0].sample(
            min(2000, len(grid_plot[grid_plot['hwy'] == 0])),
            random_state=42
        )
        ax_scatter.scatter(
            sample['dist_to_nearest_hwy'] / 1000,
            sample[prob_col],
            alpha=0.15, s=3, color='steelblue'
        )
        # rolling mean
        sorted_s = sample.sort_values('dist_to_nearest_hwy')
        window = max(50, len(sorted_s) // 20)
        rolling_mean = (sorted_s[prob_col]
                        .rolling(window, center=True, min_periods=10)
                        .mean())
        ax_scatter.plot(
            sorted_s['dist_to_nearest_hwy'].values / 1000,
            rolling_mean.values,
            color='tomato', linewidth=2, label='Rolling mean'
        )
        ax_scatter.set_xlabel('Distance to Nearest Highway Square (km)')
        ax_scatter.set_ylabel('Predicted Probability')
        ax_scatter.set_title('Probability vs. Distance to Real Highway\n'
                              '(non-hwy squares only)')
        ax_scatter.legend(fontsize=8)

    plt.suptitle(
        f'{city.title()} — Detailed CNN Probability Analysis',
        fontsize=13, fontweight='bold', y=1.01
    )

    if save_path is None:
        save_path = f'cnn/figures/probability_heatmap_{city}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'saved: {save_path}')

    return grid_plot


# --- run ---
import os
os.makedirs('cnn/figures', exist_ok=True)

# all cities overview
grid_with_probs = plot_probability_heatmap(
    preds_csv  = dataroot + filename_out,
    grid       = grid,
    threshold  = 0.05,
    save_path  = 'cnn/figures/probability_heatmap_model4.png'
)

# grid_with_probs = plot_probability_heatmap(preds_csv = model1, grid = grid, threshold = 0.05, save_path = 'cnn/figures/probability_heatmap_model1.png')
# grid_with_probs = plot_probability_heatmap(preds_csv = model2, grid = grid, threshold = 0.05, save_path = 'cnn/figures/probability_heatmap_model2.png')
# grid_with_probs = plot_probability_heatmap(preds_csv = model3, grid = grid, threshold = 0.05, save_path = 'cnn/figures/probability_heatmap_model3.png')

# detailed view per city
for city in grid['city'].unique():
    plot_probability_heatmap_detailed(
        preds_csv = model4,
        grid      = grid,
        city      = city,
        threshold = 0.05
    )