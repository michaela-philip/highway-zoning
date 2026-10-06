import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from analysis.lib.marginal_effects import Sweep, predicted_outcomes

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


# Paul Tol's vibrant blue/orange -- color-blind safe pair for the two `across` levels of an
# effect-curve plot; the difference between them is drawn in neutral black
ACROSS_COLORS = ['#0077BB', '#EE7733']
DIFF_COLOR = '#222222'


def plot_effect_curves(fit, df, comparison, sweep_var, values=None, hold=None, eval_at='ame',
                       path=None, xlabel=None, percent_x=False, title=None, ylabel=None,
                       effect_ci=None, n_whiskers=4, figsize=(5.5, 4.6)):
    """
    Plot a marginal_effects.Comparison as smooth curves over a continuous `sweep_var`:
    the top panel shows the effect (first effect level minus second) at each of the two
    `across` levels, labeled at their right ends; the bottom panel shows the difference
    between them (second across level minus first) on its own scale, with a zero line.

    95% CIs: the difference gets a shaded band. The two effects get none by default --
    their intervals typically nest inside each other and are unreadable as bands or
    outlines, and whether they overlap is the wrong test anyway (two overlapping CIs can
    still differ significantly); the bottom panel is the right test. effect_ci='whiskers'
    adds their CIs as vertical bars at `n_whiskers` evenly spaced points along the curves,
    nudged apart so the two don't overlap.

    Runs predicted_outcomes once with a Sweep over `values` (default: 21 evenly spaced
    points between the 2nd and 98th percentiles of df[sweep_var]) -- every point costs
    four predictions, so trim `values` if bootstrap draws make that slow. hold / eval_at
    are passed through. percent_x formats a 0-1 sweep variable as 0%-100%.
    Saves to `path` (use .pdf for a vector figure) and closes the figure if given;
    otherwise returns the figure.
    """
    assert effect_ci in (None, 'whiskers')
    cmp = comparison
    if sweep_var.var in (cmp.effect.var, cmp.across.var):
        raise ValueError(f"sweep_var {sweep_var.label!r} is the comparison's effect/across variable -- "
                         "the comparison overrides it, so every point would be identical")
    if values is None:
        values = np.linspace(*df[sweep_var.var].quantile([0.02, 0.98]), 21)
    values = np.asarray(values, dtype=float)
    whisker_x = np.array([])
    if effect_ci == 'whiskers':
        # evenly spaced interior positions (e.g. 20/40/60/80% of the range), added to the
        # grid so each whisker's CI is computed exactly there
        whisker_x = np.linspace(values.min(), values.max(), n_whiskers + 2)[1:-1]
        values = np.union1d(values, whisker_x)
    names = [f'x{i}' for i in range(len(values))]
    res = predicted_outcomes(fit, df, cmp, columns=Sweep(sweep_var, dict(zip(names, values))),
                             hold=hold, eval_at=eval_at, verbose=False)
    f = res.frame

    def curve(label):
        r = f[f['label'] == label].set_index('column').loc[names]
        return r['estimate'].to_numpy(), r['ci_lo'].to_numpy(), r['ci_hi'].to_numpy()

    across = list(cmp.across_levels)

    def style(ax):
        ax.set_xlim(values[0], values[-1])
        ax.axhline(0, color='#888888', lw=0.8, ls='--', zorder=1.5)
        ax.grid(axis='y', color='#EEEEEE', lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        if percent_x:
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            if values[0] >= 0 and values[-1] <= 1:
                ax.set_xticks([t for t in (0, 0.25, 0.5, 0.75, 1) if values[0] <= t <= values[-1]])

    with matplotlib.rc_context(FIG_RC):
        fig, (top, bot) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [2.2, 1], 'hspace': 0.25})
        ends = []
        for color, a in zip(ACROSS_COLORS, across):
            est, lo, hi = curve(f'{a} {cmp.effect_name}')
            top.plot(values, est, color=color, lw=2, zorder=3)
            if effect_ci == 'whiskers' and np.isfinite(lo).any():
                idx = np.searchsorted(values, whisker_x)
                nudge = (len(ends) - 0.5) * 0.012 * (values[-1] - values[0])
                xs = values[idx] + nudge
                top.vlines(xs, lo[idx], hi[idx], color=color, lw=1.2, zorder=2)
                top.plot(xs, est[idx], linestyle='none', marker='o', markersize=4, color=color, zorder=4)
            ends.append((est[-1], a))
        style(top)
        top.set_ylabel(ylabel or cmp.effect_name)
        if title:
            top.set_title(title, fontsize=10, loc='left')

        # direct labels at the right end of each curve, nudged apart if they'd collide
        y0, y1 = top.get_ylim()
        min_gap = 0.07 * (y1 - y0)
        (ya, la), (yb, lb) = sorted(ends)
        if yb - ya < min_gap:
            mid = (ya + yb) / 2
            ya, yb = mid - min_gap / 2, mid + min_gap / 2
        for y, lbl in ((ya, la), (yb, lb)):
            top.annotate(lbl, (values[-1], y), xytext=(6, 0), textcoords='offset points',
                         va='center', fontsize=9, color='#222222', annotation_clip=False)

        est, lo, hi = curve(cmp.difference_name)
        if np.isfinite(lo).any():
            bot.fill_between(values, lo, hi, color=DIFF_COLOR, alpha=0.2, lw=0)
        bot.plot(values, est, color=DIFF_COLOR, lw=2, zorder=3)
        style(bot)
        bot.set_title(f'{cmp.difference_name} ({across[1]} − {across[0]})', fontsize=10, loc='left')
        bot.set_ylabel('Difference')
        bot.set_xlabel(xlabel or sweep_var.label)
        fig.align_ylabels([top, bot])

        if path is None:
            return fig
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_comparison(result, path=None, figsize=(5, 3.4), ylabel=None):
    """
    Plot a single-column marginal_effects.PredictedOutcomes (predicted_outcomes with
    columns=None) -- the no-sweep counterpart of plot_effect_curves. Panel A shows the
    predicted outcome in each of the four cells, Panel B the effect at each `across` level
    and their difference against a zero line, all with 95% CIs (when the fit carried SEs).

    Color identifies the `across` level; in Panel A filled vs. hollow markers distinguish
    the two `effect` levels, and the difference is a black diamond, so nothing relies on
    color alone. One legend sits under the figure, keyed by those encodings.

    ylabel  Panel A y-axis label (default: 'Predicted Probability' under a logit link,
            else 'Predicted Outcome').
    Saves to `path` (use .pdf for a vector figure) and closes the figure if given;
    otherwise returns the figure.
    """
    if len(result.columns) != 1:
        raise ValueError(f"plot_comparison takes a single-column result (got {len(result.columns)} "
                         "columns) -- for a sweep use plot_effect_curves")
    cmp = result.comparison
    col = result.columns[0]
    rows = result.frame[result.frame['column'] == col].set_index('label')
    across, effect = list(cmp.across_levels), list(cmp.effect_levels)

    def point(ax, label, x, color, marker='o', filled=True):
        r = rows.loc[label]
        if np.isfinite(r['ci_lo']):
            ax.errorbar(x, r['estimate'], yerr=[[r['estimate'] - r['ci_lo']], [r['ci_hi'] - r['estimate']]],
                        fmt='none', ecolor=color, elinewidth=1.2, capsize=0, zorder=3)
        ax.plot(x, r['estimate'], linestyle='none', marker=marker, markersize=6.5, color=color,
                markerfacecolor=color if filled else 'white', markeredgewidth=1.4, zorder=4)

    def style(ax, title, ylab, n):
        ax.set_title(title, fontsize=10, loc='left')
        ax.set_ylabel(ylab)
        ax.set_xticks([])
        ax.set_xlim(-0.6, n - 0.4)
        ax.grid(axis='y', color='#E5E5E5', lw=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    with matplotlib.rc_context(FIG_RC):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)

        i = 0
        for ai, a in enumerate(across):
            for ei, e in enumerate(effect):
                point(a1, f'{a} {e}', i, ACROSS_COLORS[ai], filled=(ei == 0))
                i += 1
        if 'Sample Average' in rows.index:
            a1.axhline(rows.loc['Sample Average', 'estimate'], color='#888888', lw=1, ls=':', zorder=1)
        style(a1, 'A. Predicted Outcomes',
              ylabel or ('Predicted Probability' if result.link == 'logit' else 'Predicted Outcome'), 4)

        a2.axhline(0, color='#888888', lw=0.8, zorder=1)
        for ai, a in enumerate(across):
            point(a2, f'{a} {cmp.effect_name}', ai, ACROSS_COLORS[ai])
        point(a2, cmp.difference_name, 2, DIFF_COLOR, marker='D')
        style(a2, f'B. {cmp.effect_name}', cmp.effect_name, 3)

        def key(color, marker='o', filled=True, **kw):
            return Line2D([], [], linestyle='none', marker=marker, markersize=6.5, color=color,
                          markerfacecolor=color if filled else 'white', markeredgewidth=1.4, **kw)
        handles = [key(ACROSS_COLORS[i], label=a) for i, a in enumerate(across)]
        handles += [key('#777777', filled=(i == 0), label=e) for i, e in enumerate(effect)]
        if 'Sample Average' in rows.index:
            handles.append(Line2D([], [], color='#888888', lw=1, linestyle=':', label='Sample Average'))
        handles.append(key(DIFF_COLOR, 'D', label=f'{cmp.difference_name} ({across[1]} − {across[0]})'))
        fig.tight_layout()
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3,
                   frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.2)

        if path is None:
            return fig
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
