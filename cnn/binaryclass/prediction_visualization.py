import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import glob
import os

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

grid = pd.read_pickle('data/output/sample.pkl')
dataroot = 'cnn/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv_files.sort(key=os.path.getmtime, reverse=True)
last_csv = csv_files[0]
filename_out = os.path.basename(last_csv)
print(f'Using most recent predictions from: {filename_out}')

# run it — point to your most recent prediction csv
# plot_propensity_overlap(
#     results_csv_path=dataroot + filename_out,  # uses the filename from your prediction script
#     grid=grid,
#     temperature_values=[1.0, 5.0, 10.0, 20.0]
# )

# --- run it ---
plot_highway_recall(
    results_csv_path=dataroot + filename_out,
    grid=grid,
    temperature_values=[1.0, 5.0, 10.0, 20.0]
)