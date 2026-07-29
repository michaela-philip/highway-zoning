import pandas as pd
import re
from types import SimpleNamespace
import numpy as np


def format_regression_results(results):
    """Format a fitted results object (a raw statsmodels result, or the SimpleNamespace
    produced by bootstrap_results_to_namespace) into a LaTeX-ready single-column
    DataFrame: one row per coefficient (stacked coef^{stars} over (SE)), plus
    R-squared/Observations rows. Assumes results.params/.bse/.pvalues are already indexed
    by friendly display labels (true for everything produced via analysis.lib.specs.build_spec
    and analysis.lib.bootstrap's fit functions)."""
    df = pd.DataFrame({'coef':results.params, 'stderror': results.bse, 'pvalue': results.pvalues})[1:]
    def sig_coef(row):
        if row['pvalue'] < 0.001:
            return f"{row['coef']:.3f}^{{***}}"
        elif row['pvalue'] < 0.01:
            return f"{row['coef']:.3f}^{{**}}"
        elif row['pvalue'] < 0.05:
            return f"{row['coef']:.3f}^{{*}}"
        else:
            return f"{row['coef']:.3f}"
    df['Coefficient'] = df.apply(
        lambda row: f"\\makecell[tr]{{{sig_coef(row)} \\\\ ({row['stderror']:.3f})}}", axis=1)
    df = df[['Coefficient']]
    df.loc['R-squared'] = [f"{results.rsquared:.3f}"]
    df.loc['Observations'] = [f"{int(results.nobs)}"]
    return df


def _wrap_threeparttable(text, widthmultiplier, notes=None):
    """Swap the Styler-generated `tabular` for a fixed-width `tabular*`, wrapped in
    `threeparttable` so the notes box is sized to the table's own width rather than the
    full text width. `notes` is an optional string or list of strings, each rendered as
    an \\item line in a \\begin{tablenotes} block just below the table.

    Requires \\usepackage{threeparttable} (and \\usepackage{makecell}, already needed for
    the coefficient cells) in the including document's preamble -- this function only
    emits the table fragment, not the preamble.
    """
    text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}')
    text = text.replace('\\end{tabular}', '\\end{tabular*}')
    text = text.replace('\\begin{tabular*}', '\\begin{threeparttable}\n\\begin{tabular*}')

    notes_block = ''
    if notes:
        if isinstance(notes, str):
            notes = [notes]
        items = '\n'.join(f'\\item {note}' for note in notes)
        notes_block = f'\\begin{{tablenotes}}\n\\small\n{items}\n\\end{{tablenotes}}\n'
    text = text.replace('\\end{tabular*}', f'\\end{{tabular*}}\n{notes_block}\\end{{threeparttable}}')
    return text


def _write_latex_table(df, caption, label, widthmultiplier, notes):
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2).to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = _wrap_threeparttable(text, widthmultiplier, notes)
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)


# table with one regression - no concatenating
def export_single_regression(df, caption, label, widthmultiplier = 1.0, leaveout = None, notes = None):
    df = df.drop(index=leaveout, errors='ignore') if leaveout is not None else df
    _write_latex_table(df, caption, label, widthmultiplier, notes)


# table with multiple regressions - definition of 'Black' as column title
def export_multiple_regressions(df_dict, caption, label, leaveout = None, widthmultiplier = 1.0, notes = None):
    renamed_list = [df.rename(columns = {'Coefficient': title}) for title, df in df_dict.items()]
    df = pd.concat(renamed_list, axis = 1)
    df = df.drop(index=leaveout, errors='ignore') if leaveout is not None else df
    _write_latex_table(df, caption, label, widthmultiplier, notes)
