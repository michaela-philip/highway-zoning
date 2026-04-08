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

    # --- Plot 2: temperature scaling experiments ---
    # we need the raw logits to apply temperature, so back-transform via logit
    # prob = sigmoid(logit) => logit = log(p / (1-p))
    eps = 1e-6
    raw_probs = preds['prob_hwy'].values.clip(eps, 1 - eps)
    logits = np.log(raw_probs / (1 - raw_probs))

    n_temps = len(temperature_values)
    fig, axes = plt.subplots(n_temps, 2, figsize=(14, 4 * n_temps))
    if n_temps == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Propensity Score Overlap — Temperature Scaling', fontsize=12)

    for row_idx, T in enumerate(temperature_values):
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        
        sp_hwy1 = scaled_probs[preds['hwy'] == 1]
        sp_hwy0 = scaled_probs[preds['hwy'] == 0]

        overlap_mask = (scaled_probs >= 0.1) & (scaled_probs <= 0.9)
        overlap_frac = overlap_mask.mean()

        print(f'\nT={T:.1f} | mean(hwy=1)={sp_hwy1.mean():.3f} | mean(hwy=0)={sp_hwy0.mean():.3f} | '
              f'frac in [0.1, 0.9]: {100*overlap_frac:.1f}%')

        for col_idx, yscale in enumerate(['linear', 'log']):
            ax = axes[row_idx, col_idx]
            ax.hist(sp_hwy0, bins=100, alpha=0.5, density=True, label='highway=0', color='steelblue')
            ax.hist(sp_hwy1, bins=100, alpha=0.5, density=True, label='highway=1', color='tomato')
            ax.set_xlabel('scaled probability')
            ax.set_ylabel('density')
            ax.set_yscale(yscale)
            ax.legend(fontsize=8)
            ax.set_title(f'T={T:.1f}, scale={yscale}')
            # mark the overlap region
            ax.axvline(0.1, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axvline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.tight_layout()
    plt.savefig('cnn/figures/overlap_temperature.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('saved: cnn/figures/overlap_temperature.png')

    # --- Plot 3: common support summary across temperatures ---
    fig, ax = plt.subplots(figsize=(8, 4))
    thresholds = [0.05, 0.10, 0.20, 0.30]
    for T in temperature_values:
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        overlap_fracs = [((scaled_probs >= t) & (scaled_probs <= 1-t)).mean() for t in thresholds]
        ax.plot([f'[{t},{1-t}]' for t in thresholds], overlap_fracs, marker='o', label=f'T={T}')
    ax.set_xlabel('overlap region')
    ax.set_ylabel('fraction of cells in region')
    ax.set_title('Common Support by Temperature and Threshold')
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig('cnn/figures/overlap_support.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('saved: cnn/figures/overlap_support.png')

# run it — point to your most recent prediction csv
# plot_propensity_overlap(
#     results_csv_path=dataroot + filename_out,  # uses the filename from your prediction script
#     grid=grid,
#     temperature_values=[1.0, 5.0, 10.0, 20.0]
# )

def plot_highway_recall(results_csv_path, grid, temperature_values=[1.0, 5.0, 10.0, 20.0]):
    # --- load and merge (same as your existing function) ---
    preds = pd.read_csv(results_csv_path)
    hwy_labels = grid[['grid_id', 'hwy']].drop_duplicates('grid_id')
    preds = preds.merge(hwy_labels, on='grid_id', how='left')
    preds['hwy'] = preds['hwy'].fillna(0).astype(int)

    eps = 1e-6
    raw_probs = preds['prob_hwy'].values.clip(eps, 1 - eps)
    logits = np.log(raw_probs / (1 - raw_probs))

    thresholds = np.linspace(0.01, 0.95, 200)

    n_temps = len(temperature_values)
    fig, axes = plt.subplots(n_temps, 2, figsize=(14, 4 * n_temps))
    if n_temps == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle('Highway Recall and False Inclusion Rate by Threshold', fontsize=13)

    for row_idx, T in enumerate(temperature_values):
        scaled_probs = 1 / (1 + np.exp(-logits / T))

        hwy1_probs = scaled_probs[preds['hwy'] == 1]
        hwy0_probs = scaled_probs[preds['hwy'] == 0]
        n_hwy1 = len(hwy1_probs)
        n_hwy0 = len(hwy0_probs)

        # For each threshold: fraction of hwy=1 squares ABOVE threshold (recall)
        recall       = [(hwy1_probs >= t).mean() for t in thresholds]
        # For each threshold: fraction of hwy=0 squares ABOVE threshold (false inclusion)
        false_incl   = [(hwy0_probs >= t).mean() for t in thresholds]
        # Total squares retained above threshold
        total_retain = [(scaled_probs >= t).mean() for t in thresholds]

        # --- Left panel: recall and false inclusion rate ---
        ax = axes[row_idx, 0]
        ax.plot(thresholds, recall,      color='tomato',    linewidth=1.8, label='Recall (hwy=1 retained)')
        ax.plot(thresholds, false_incl,  color='steelblue', linewidth=1.8, label='False inclusion (hwy=0 retained)')
        ax.plot(thresholds, total_retain,color='gray',      linewidth=1.2, linestyle='--', label='All cells retained')

        # Mark common thresholds
        for tau in [0.10, 0.20, 0.30]:
            r   = (hwy1_probs >= tau).mean()
            fi  = (hwy0_probs >= tau).mean()
            ax.axvline(tau, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.annotate(f'τ={tau}\nrecall={r:.2f}\nFI={fi:.2f}',
                        xy=(tau, r), xytext=(tau + 0.02, r - 0.15),
                        fontsize=7, color='black',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

        ax.set_xlabel('Threshold τ')
        ax.set_ylabel('Fraction of squares retained')
        ax.set_title(f'T={T:.1f} — Recall vs False Inclusion')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # --- Right panel: precision-recall style tradeoff ---
        # x = false inclusion rate, y = recall — analogous to ROC curve
        ax2 = axes[row_idx, 1]
        ax2.plot(false_incl, recall, color='darkorange', linewidth=1.8)

        # Mark the same reference thresholds on the ROC-style curve
        for tau in [0.10, 0.20, 0.30]:
            r  = (hwy1_probs >= tau).mean()
            fi = (hwy0_probs >= tau).mean()
            ax2.scatter(fi, r, color='black', zorder=5, s=30)
            ax2.annotate(f'τ={tau}', xy=(fi, r), xytext=(fi + 0.01, r - 0.04), fontsize=7)

        ax2.set_xlabel('False inclusion rate (hwy=0 retained)')
        ax2.set_ylabel('Recall (hwy=1 retained)')
        ax2.set_title(f'T={T:.1f} — Recall vs False Inclusion Tradeoff')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
        ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

        # Print summary stats for this temperature
        print(f'\nT={T:.1f}:')
        for tau in [0.05, 0.10, 0.20, 0.30]:
            r  = (hwy1_probs >= tau).mean()
            fi = (hwy0_probs >= tau).mean()
            n  = (scaled_probs >= tau).sum()
            print(f'  τ={tau:.2f} | recall={r:.3f} | false inclusion={fi:.3f} | n retained={n:,}')

    plt.tight_layout()
    plt.savefig('cnn/figures/recall_diagnostics.png', dpi=100, bbox_inches='tight')
    plt.show()
    print('saved: cnn/figures/recall_diagnostics.png')

# # --- run it ---
# plot_highway_recall(
#     results_csv_path=dataroot + filename_out,
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

from scipy.optimize import minimize_scalar
from scipy.special import expit

def nll_temperature(T, logits, labels):
    """Negative log likelihood of labels given temperature-scaled logits"""
    scaled_probs = expit(logits / T)
    scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
    return -np.mean(labels * np.log(scaled_probs) + 
                    (1 - labels) * np.log(1 - scaled_probs))

preds = pd.read_csv(dataroot + filename_out)
hwy_labels = grid[['grid_id', 'hwy']].drop_duplicates('grid_id')
preds = preds.merge(hwy_labels, on='grid_id', how='left')
preds['hwy'] = preds['hwy'].fillna(0).astype(int)

eps = 1e-6
raw_probs = preds['prob_hwy'].values.clip(eps, 1 - eps)
logits = np.log(raw_probs / (1 - raw_probs))
labels = preds['hwy'].values

# result = minimize_scalar(
#     nll_temperature, 
#     bounds=(0.1, 50.0), 
#     method='bounded',
#     args=(logits, labels)
# )
# optimal_T = result.x
# print(f'Optimal temperature: {optimal_T:.3f}')

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


# --- run it ---
smd_df, smd_matrix = plot_geographic_similarity_conditional(
    dataroot + filename_out,
    grid,
    geo_features = ['elevation', 'dist_water', 'dist_to_hwy', 'distance_to_cbd', 'dist_to_rr', 'flood_risk'],
    prob_bins  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
)