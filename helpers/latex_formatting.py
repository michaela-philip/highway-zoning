import pandas as pd
import re
from types import SimpleNamespace
import numpy as np


def _extract_rsquared(results, r2_label):
    """(value, label) for the R^2 row, or (None, label) to skip it. Prefers
    results.rsquared (OLS, or a SimpleNamespace that already set one -- e.g.
    bootstrap_results_to_namespace, or bootstrap_ppml_table overriding it with a
    deviance-based pseudo R^2). Falls back to results.pseudo_rsquared('cs') -- present
    on an out-of-the-box statsmodels GLM result (e.g. sm.GLM(...).fit()), which has no
    .rsquared at all. `r2_label`, if given, always wins; otherwise defaults to
    'R-squared' or 'Pseudo R-squared' to match whichever value was actually used."""
    rsq = getattr(results, 'rsquared', None)
    if rsq is not None:
        return rsq, (r2_label or 'R-squared')
    pseudo_rsquared = getattr(results, 'pseudo_rsquared', None)
    if callable(pseudo_rsquared):
        try:
            return pseudo_rsquared('cs'), (r2_label or 'Pseudo R-squared')
        except Exception:
            pass
    return None, r2_label


def format_regression_results(results, r2_label=None, x_vars=None, columns=None):
    """Format a fitted results object into a LaTeX-ready single-column DataFrame: one row
    per coefficient (stacked coef^{stars} over (SE)), plus R^2/Observations rows.

    `results` just needs .params/.bse/.pvalues/.nobs (see _extract_rsquared above for the
    R^2 row). That covers:
      - the SimpleNamespace produced by bootstrap_results_to_namespace or returned by
        analysis.lib.standard_errors.fit_ppml_conley, and
      - an out-of-the-box statsmodels result, e.g. sm.GLM(...).fit() or sm.OLS(...).fit().

    By default results.params/.bse/.pvalues are assumed already indexed by friendly
    display labels with an 'Intercept'/'const' entry (true for everything produced via
    analysis.lib.bootstrap's fit functions and fit_ppml_conley) -- that entry is dropped
    automatically, wherever it falls.

    Pass x_vars/columns (as returned by analysis.lib.specs.build_spec) when `results`
    instead comes from a model fit directly on df[x_vars] -- e.g.
    sm.GLM(df['hwy'], df[x_vars], ...).fit(cov_type='HC3') -- so results.params is
    indexed by the raw x_var names instead. They're relabeled to the friendly columns[1:]
    labels before formatting. Works whether or not that design matrix had an intercept
    added (sm.add_constant) -- an intercept/'const' row, if present, is still dropped."""
    params, bse, pvalues = pd.Series(results.params), pd.Series(results.bse), pd.Series(results.pvalues)

    if x_vars is not None and columns is not None:
        rename = dict(zip(x_vars, columns[1:]))
        params, bse, pvalues = (s.rename(index=rename) for s in (params, bse, pvalues))

    df = pd.DataFrame({'coef': params, 'stderror': bse, 'pvalue': pvalues})
    df = df.drop(index=[i for i in ('Intercept', 'const') if i in df.index])

    def sig_coef(row):
        if row['pvalue'] < 0.001:
            return f"{row['coef']:.3f}{{***}}"
        elif row['pvalue'] < 0.01:
            return f"{row['coef']:.3f}{{**}}"
        elif row['pvalue'] < 0.05:
            return f"{row['coef']:.3f}{{*}}"
        else:
            return f"{row['coef']:.3f}"
    df['Coefficient'] = df.apply(
        lambda row: f"\\makecell[tr]{{{sig_coef(row)} \\\\ ({row['stderror']:.3f})}}", axis=1)
    df = df[['Coefficient']]

    rsq, r2_label = _extract_rsquared(results, r2_label)
    if rsq is not None:
        df.loc[r2_label] = [f"{rsq:.3f}"]
    df.loc['Observations'] = [f"{int(results.nobs)}"]
    return df


def _wrap_threeparttable(text, widthmultiplier, notes=None, long=False):
    """Swap the Styler-generated `tabular` for a fixed-width `tabular*`, wrapped in
    `threeparttable` so the notes box is sized to the table's own width rather than the
    full text width. `notes` is an optional string or list of strings, each rendered as
    an \\item line in a \\begin{tablenotes} block just below the table.

    Requires \\usepackage{threeparttable} (and \\usepackage{makecell}, already needed for
    the coefficient cells) in the including document's preamble -- this function only
    emits the table fragment, not the preamble.

    Pass long=True when `text` was generated with to_latex(environment='longtable', ...)
    -- for tables too tall for one page. Wraps it in threeparttablex's ThreePartTable/
    TableNotes instead, which (unlike plain threeparttable) is longtable-compatible.
    Requires \\usepackage{longtable} and \\usepackage{threeparttablex} in the preamble;
    widthmultiplier is ignored on this path since longtable's width is already fixed by
    its own column_format.
    """
    if long:
        notes_block = ''
        if notes:
            if isinstance(notes, str):
                notes = [notes]
            items = '\n'.join(f'\\item {note}' for note in notes)
            notes_block = f'\\begin{{TableNotes}}[flushleft]\n\\footnotesize\n{items}\n\\end{{TableNotes}}\n'
        text = text.replace('\\begin{longtable}',
                             f'\\begin{{ThreePartTable}}\n{notes_block}\\begin{{longtable}}')
        text = text.replace('\\end{longtable}',
                             '\\insertTableNotes\n\\end{longtable}\n\\end{ThreePartTable}')
        return text

    text = text.replace('\\begin{tabular}', f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}')
    text = text.replace('\\end{tabular}', '\\end{tabular*}')
    text = text.replace('\\begin{tabular*}', '\\begin{threeparttable}\n\\begin{tabular*}')

    notes_block = ''
    if notes:
        if isinstance(notes, str):
            notes = [notes]
        items = '\n'.join(f'\\item {note}' for note in notes)
        notes_block = f'\\begin{{tablenotes}}[flushleft]\n\\footnotesize\n{items}\n\\end{{tablenotes}}\n'
    text = text.replace('\\end{tabular*}', f'\\end{{tabular*}}\n{notes_block}\\end{{threeparttable}}')
    return text


def export_table(df, caption, label, widthmultiplier = 1.0, notes = None):
    """Export an arbitrary already-formatted DataFrame (any columns, not just regression
    results) as a threeparttable/tabular* LaTeX table -- the same formatting/export
    machinery export_single_regression/export_multiple_regressions use internally, exposed
    directly for tables that aren't per-variable regression coefficients (e.g. descriptive
    summary statistics)."""
    num_cols = df.shape[1]
    col_format = '@{\\extracolsep{\\fill}}l*' + f'{{{num_cols}}}' + '{r}'
    text = df.style.format(precision=2, na_rep = '').to_latex(position_float = 'centering',
                caption=caption, position = 'h', label=label, hrules=True, column_format = col_format)
    text = _wrap_threeparttable(text, widthmultiplier, notes)
    filename = label.split(':')[-1] + '.tex'
    with open('tables/' + filename, 'w') as f:
        f.write(text)


# table with one regression - no concatenating
def export_single_regression(df, caption, label, widthmultiplier = 1.0, leaveout = None, notes = None):
    df = df.drop(index=leaveout, errors='ignore') if leaveout is not None else df
    export_table(df, caption, label, widthmultiplier, notes)


# table with multiple regressions - definition of 'Black' as column title
def export_multiple_regressions(df_dict, caption, label, leaveout = None, widthmultiplier = 1.0, notes = None):
    renamed_list = [df.rename(columns = {'Coefficient': title}) for title, df in df_dict.items()]
    df = pd.concat(renamed_list, axis = 1)
    df = df.reindex([i for i in df.index if i not in ('R-squared', 'Observations')] + ['R-squared', 'Observations'])
    df = df.drop(index=leaveout, errors='ignore') if leaveout is not None else df
    export_table(df, caption, label, widthmultiplier, notes)
