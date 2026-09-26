import os

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
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


def plot_access_map(sample, grid, value_col, path, colorbar_label,
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
        grid[grid['hwy_40'] == 1].plot(ax=ax, color=HWY40_GRAY, edgecolor='face', linewidth=0.3)
        ax.set_axis_off()
        ax.set_aspect('equal')
        if scale_bar_m:
            _scale_bar(ax, scale_bar_m, f'{scale_bar_m / 1000:g} km')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02,
                            aspect=35, extend=extend)
        cbar.set_label(colorbar_label)
        cbar.outline.set_linewidth(0.5)

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
