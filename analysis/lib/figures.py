import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Paul Tol's sequential YlOrBr scheme (https://personal.sron.nl/~pault/) -- color-blind
# safe, prints well in grayscale, and matches the Tol colors used in the paper.
TOL_YLORBR = ['#FFFFE5', '#FFF7BC', '#FEE391', '#FEC44F', '#FB9A29',
              '#EC7014', '#CC4C02', '#993404', '#662506']
# neutral grays for context layers, distinct in lightness from every YlOrBr step
EXCLUDED_GRAY = '#DDDDDD'
HWY40_GRAY = '#555555'

# serif text so the figure sits naturally next to LaTeX body text (no usetex -- it
# needs a TeX install wherever this runs)
FIG_RC = {
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'pdf.fonttype': 42,
}


def _scale_bar(ax, length_m, label):
    """Plain black scale bar in the lower-left corner; assumes a projected CRS in meters
    (true for the sample grid -- compute_characteristic_access measures distance in m)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.04 * (x1 - x0)
    y = y0 + 0.04 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color='black', lw=2, solid_capstyle='butt')
    ax.text(x + length_m / 2, y + 0.015 * (y1 - y0), label, ha='center', va='bottom', fontsize=9)


def plot_access_map(sample, grid, value_col, path, colorbar_label, hwy_40 = True, hwy = False,
                    clip_quantiles=(0.01, 0.99), scale_bar_m=2000, figsize=(6.5, 6.5),
                    excluded_label='Not in sample'):
    """
    Choropleth of `value_col` over the grid squares in `sample` (one city), drawn on top
    of that city's full `grid` so any squares missing from the sample show up as gray
    (labeled `excluded_label`) rather than as holes, with pre-1940 highway squares
    (grid['hwy_40'] == 1) drawn over the top in dark gray for orientation.

    The color scale is clipped at `clip_quantiles` of value_col (with arrows on the
    colorbar) so a handful of extreme squares don't wash out the rest of the map; pass
    None to use the full range. Saves to `path` -- use .pdf for a vector figure.
    """
    sample = gpd.GeoDataFrame(sample, geometry='geometry', crs=grid.crs)
    vals = sample[value_col]
    if clip_quantiles is None:
        vmin, vmax, extend = vals.min(), vals.max(), 'neither'
    else:
        vmin, vmax = vals.quantile(list(clip_quantiles))
        extend = 'both'
    cmap = LinearSegmentedColormap.from_list('tol_ylorbr', TOL_YLORBR)
    norm = Normalize(vmin=vmin, vmax=vmax)

    with matplotlib.rc_context(FIG_RC):
        fig, ax = plt.subplots(figsize=figsize)
        excluded = grid[~grid['grid_id'].isin(sample['grid_id'])]
        if len(excluded):
            excluded.plot(ax=ax, color=EXCLUDED_GRAY, edgecolor='face', linewidth=0.3)
        sample.plot(ax=ax, column=value_col, cmap=cmap, norm=norm, edgecolor='face', linewidth=0.3)
        # pre-1940 highways drawn on top, for orientation
        if hwy_40 == True:
            grid[grid['hwy_40'] == 1].plot(ax=ax, color=HWY40_GRAY, edgecolor='face', linewidth=0.3)
        if hwy == True:
            grid[grid['hwy'] == 1].plot(ax=ax, color=HWY40_GRAY, edgecolor='face', linewidth=0.3)
        ax.set_axis_off()
        ax.set_aspect('equal')
        if scale_bar_m:
            _scale_bar(ax, scale_bar_m, f'{scale_bar_m / 1000:g} km')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02,
                            aspect=35, extend=extend)
        cbar.set_label(colorbar_label)
        cbar.outline.set_linewidth(0.5)

        if hwy_40 == True:
            handles = [Patch(facecolor=HWY40_GRAY, label='Pre-1940 highway')]
            if len(excluded):
                handles.append(Patch(facecolor=EXCLUDED_GRAY, label=excluded_label))
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1), ncol=2, frameon=False, fontsize=9)

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def export_figure_tex(image_path, caption, label, notes=None, width=r'\textwidth'):
    """Write a LaTeX figure environment (to tables/<label suffix>.tex, alongside the
    tables) that \\includegraphics the saved image, with optional notes beneath it in
    the same footnotesize style as the tables. Needs graphicx in the preamble;
    image_path is written as given, so make it relative to the main .tex file."""
    lines = ['\\begin{figure}[h]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             f'\\includegraphics[width={width}]{{{image_path}}}']
    if notes:
        lines += ['\\par\\vspace{0.5em}', f'\\begin{{minipage}}{{{width}}}',
                  f'\\footnotesize {notes}', '\\end{minipage}']
    lines.append('\\end{figure}')
    with open('tables/' + label.split(':')[-1] + '.tex', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# Paul Tol's vibrant blue/orange -- color-blind safe pair for the two `across` levels of a
# predicted-outcomes plot; the difference between them is drawn in neutral black
ACROSS_COLORS = ['#0077BB', '#EE7733']
DIFF_COLOR = '#222222'


def plot_predicted_outcomes(result, path=None, figsize=(7.5, 3.4), panels=('cells', 'effects'),
                            column_group=None, ylabel=None, connect=None):
    """
    Plot a marginal_effects.PredictedOutcomes as a figure that mirrors
    export_predicted_outcomes_table: Panel A ('cells') the predicted outcome for each of the
    four cells, Panel B ('effects') the effect at each `across` level and their difference,
    each with its 95% CI (when the fit carried SEs). The table's columns (Sweep levels,
    Strata bins, or the single 'Estimate') run along the x-axis; points at the same column
    are dodged so their CIs don't overlap.

    Color identifies the `across` level; within Panel A, filled vs. hollow markers
    distinguish the two `effect` levels, so nothing relies on color alone.

    panels        which panels to draw, in order -- ('cells',), ('effects',), or both.
    column_group  x-axis title (default: the Sweep/Strata title).
    ylabel        Panel A y-axis label (default: 'Predicted Probability' under a logit
                  link, else 'Predicted Outcome').
    connect       join points across columns with lines (default: only when there's
                  more than one column).
    Saves to `path` (use .pdf for a vector figure) and closes the figure if given;
    otherwise returns the figure.
    """
    cmp = result.comparison
    f = result.frame
    cols = list(result.columns)
    across, effect = list(cmp.across_levels), list(cmp.effect_levels)
    lookup = {(r['column'], r['label']): r for _, r in f.iterrows()}
    connect = len(cols) > 1 if connect is None else connect

    headers, xtitle = cols, column_group or result.column_title
    # '10th Percentile', ... -> '10th', ... under a '<title> (Percentile)' axis label
    if xtitle and all(h.endswith(' Percentile') for h in headers):
        headers = [h[:-len(' Percentile')] for h in headers]
        xtitle = f'{xtitle} (Percentile)'
    x = np.arange(len(cols))

    def series(ax, key, offset, color, marker, filled):
        rows = [lookup[(c, key)] for c in cols]
        est = np.array([r['estimate'] for r in rows])
        lo = np.array([r['ci_lo'] for r in rows])
        hi = np.array([r['ci_hi'] for r in rows])
        xs = x + offset
        if connect:
            ax.plot(xs, est, color=color, lw=1.2, alpha=0.6, zorder=2)
        if np.isfinite(lo).any():
            ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt='none', ecolor=color,
                        elinewidth=1.2, capsize=0, zorder=3)
        ax.plot(xs, est, linestyle='none', marker=marker, markersize=6.5, color=color,
                markerfacecolor=color if filled else 'white', markeredgewidth=1.4, zorder=4)

    def style(ax, title):
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels(headers if len(cols) > 1 else [''])
        pad = 0.5 if len(cols) > 1 else 0.35
        ax.set_xlim(-pad, len(cols) - 1 + pad)
        if xtitle and len(cols) > 1:
            ax.set_xlabel(xtitle)
        ax.grid(axis='y', color='#E5E5E5', lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='x', length=0)

    letters = iter('ABC')
    with matplotlib.rc_context(FIG_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False)
        for ax, panel in zip(axes[0], panels):
            if panel == 'cells':
                width = 0.12
                offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
                i = 0
                for ai, a in enumerate(across):
                    for ei, e in enumerate(effect):
                        series(ax, f'{a} {e}', offsets[i], ACROSS_COLORS[ai], 'o', ei == 0)
                        i += 1
                if (f['kind'] == 'reference').any():
                    ref = [lookup[(c, 'Sample Average')]['estimate'] for c in cols]
                    ax.hlines(ref, x - 0.4, x + 0.4, color='#888888', lw=1, linestyles=':', zorder=1)
                ax.set_ylabel(ylabel or ('Predicted Probability' if result.link == 'logit'
                                         else 'Predicted Outcome'))
                style(ax, f'{next(letters)}. Predicted Outcomes')
            elif panel == 'effects':
                width = 0.15
                ax.axhline(0, color='#888888', lw=0.8, zorder=1)
                for ai, a in enumerate(across):
                    series(ax, f'{a} {cmp.effect_name}', (ai - 1) * width, ACROSS_COLORS[ai],
                           'o', True)
                series(ax, cmp.difference_name, width, DIFF_COLOR, 'D', True)
                ax.set_ylabel(cmp.effect_name)
                style(ax, f'{next(letters)}. {cmp.effect_name}')
            else:
                raise ValueError(f"unknown panel {panel!r} -- use 'cells' or 'effects'")

        # one legend under the figure, keyed by encoding (color = across level, fill =
        # effect level) rather than one entry per series
        def key(color, marker='o', filled=True, **kw):
            return Line2D([], [], linestyle='none', marker=marker, markersize=6.5, color=color,
                          markerfacecolor=color if filled else 'white', markeredgewidth=1.4, **kw)
        handles = [key(ACROSS_COLORS[i], label=a) for i, a in enumerate(across)]
        if 'cells' in panels:
            handles += [key('#777777', filled=(i == 0), label=e) for i, e in enumerate(effect)]
            if (f['kind'] == 'reference').any():
                handles.append(Line2D([], [], color='#888888', lw=1, linestyle=':', label='Sample Average'))
        if 'effects' in panels:
            handles.append(key(DIFF_COLOR, 'D', label=f'{cmp.difference_name} ({across[1]} − {across[0]})'))
        fig.tight_layout()
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 4), frameon=False, fontsize=8.5,
                   handletextpad=0.3, columnspacing=1.2)

        if path is None:
            return fig
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
