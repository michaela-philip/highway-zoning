from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from analysis.lib.specs import Interaction, Transform, Var, resolve_level

# Predicted outcomes for a 2 x 2 comparison: an `effect` variable evaluated at two levels
# (the effect is first level minus second), `across` two levels of a second variable, and
# the difference between the two effects (second `across` level's minus first's). E.g.
#
#   PROTECTION = Comparison(effect=RES, across=DEM_ACCESS,
#                           effect_name='Protection Effect', difference_name='Disparate Protection')
#   res = predicted_outcomes(fit, df, PROTECTION,
#                            columns=Sweep(SUIT, percentile_levels([10, 25, 50, 75, 90])))
#
# Each prediction is made by overriding base variables (Vars) and rebuilding the design
# matrix through fit.spec, so every interaction / spline term built on them is recomputed
# consistently -- there are no interaction lists to pass in. Any Var not assigned by the
# comparison, the columns, or `hold` stays at each row's own value.
#
# Levels are {name: value} dicts in order; a value can be a number or a specs.Q quantile,
# resolved on the full `df` passed to predicted_outcomes (before any Strata split). A
# continuous Var can be reported at two representative levels this way (e.g. the 25th/75th
# percentile of an exposure measure) -- keep them within the range the model actually saw,
# and name them so it's clear they're representative levels rather than two groups
# partitioning the sample.


@dataclass
class Comparison:
    effect: Var
    across: Var
    effect_levels: dict = None    # default: effect.levels
    across_levels: dict = None    # default: across.levels
    effect_name: str = 'Effect'
    difference_name: str = 'Difference'

    def __post_init__(self):
        self.effect_levels = self.effect_levels or getattr(self.effect, 'levels', None)
        self.across_levels = self.across_levels or getattr(self.across, 'levels', None)
        for role, var, levels in (('effect', self.effect, self.effect_levels),
                                  ('across', self.across, self.across_levels)):
            if not isinstance(var, Var):
                raise TypeError(f"{role} must be a Var (a base column) -- got {var!r}. To compare "
                                "levels of a transformed variable, use its parent Var.")
            if levels is None or len(levels) != 2:
                raise ValueError(f"{role} ({var.label}) needs exactly 2 levels -- pass "
                                 f"{role}_levels={{name: value, name: value}}")
        if self.effect.var == self.across.var:
            raise ValueError("effect and across must be different variables")


@dataclass
class Sweep:
    """Table columns that each set `var` to one of `levels` ({name: value}) for every row."""
    var: Var
    levels: dict
    title: str = None             # spanning header; default var.label


@dataclass
class Strata:
    """Table columns that each restrict to the rows whose `var` falls in one bin -- no row's
    covariates are rewritten. bins: an int (equal-frequency, pd.qcut) or explicit edges
    (pd.cut)."""
    var: Var
    bins: object
    labels: list = None
    title: str = None


@dataclass
class PredictedOutcomes:
    """Output of predicted_outcomes(). `frame` has one row per (column, estimate): kind is
    'cell', 'reference', 'effect' or 'difference'; `across`/`effect` name the levels
    involved; then estimate / se / p / ci_lo / ci_hi (se etc. NaN when the fit carries
    neither a covariance nor bootstrap draws) and n (rows averaged over)."""
    frame: pd.DataFrame
    comparison: Comparison
    columns: list
    column_title: str
    eval_at: str
    link: str

    def table(self, stat='estimate'):
        """label x column pivot of one stat, in display order."""
        f = self.frame
        return f.pivot(index='label', columns='column', values=stat).loc[
            f['label'].unique(), self.columns]


# --------------------------------------------------------------------------
# computation
# --------------------------------------------------------------------------

def _depends_on(term, var):
    if isinstance(term, Var):
        return term.var == var
    if isinstance(term, Transform):
        return _depends_on(term.parent, var)
    if isinstance(term, Interaction):
        return any(_depends_on(p, var) for p in term.parts())
    return False


def _check_in_spec(spec, var, role):
    if not any(_depends_on(t, var.var) for t in spec.terms):
        raise ValueError(f"{role} {var.label!r} ({var.var}) doesn't enter this model, so setting "
                         "it can't change any prediction")


def _inverse_link(eta, link):
    if link == 'log':
        with np.errstate(over='ignore'):
            return np.exp(eta)
    if link == 'logit':
        return expit(eta)
    return eta


def _cell_stats(X, fit, chunk=64):
    """(prediction, gradient wrt beta or None, bootstrap draws or None) of mean_i(f(x_i'b))
    over the rows of X. One pass per cell; every reported estimate is a linear combination
    of these.

    The matmuls spuriously raise divide-by-zero/overflow RuntimeWarnings on some BLAS
    backends (observed with Accelerate on macOS) whenever a column is all-zero, with no
    actual NaN/Inf in the result -- np.errstate suppresses that false positive."""
    link = fit.link
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        mu = _inverse_link(X @ fit.beta, link)
        pred = float(mu.mean())

        grad = None
        if fit.cov is not None:
            if link == 'identity':
                grad = X.mean(axis=0)
            else:
                dmu = mu if link == 'log' else mu * (1 - mu)
                grad = (dmu[:, None] * X).mean(axis=0)

        draws = None
        if fit.boot is not None:
            if link == 'identity':
                draws = fit.boot @ X.mean(axis=0)
            else:  # chunked so the (n, draws) matrix stays small
                draws = np.concatenate([_inverse_link(X @ fit.boot[i:i + chunk].T, link).mean(axis=0)
                                        for i in range(0, len(fit.boot), chunk)])
    return pred, grad, draws


def _cell_ci(pred, se, link):
    """95% CI for a single predicted level. Under a log/logit link the plain symmetric
    interval ignores that the prediction is bounded (>0, or in (0, 1)) and routinely
    crosses the bound for thin cells; building it on the log / log-odds scale (delta-method
    step se/pred, se/(pred(1-pred))) and transforming back keeps it in range."""
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        if link == 'log' and pred > 0:
            lo, hi = np.exp(np.log(pred) + np.array([-1.96, 1.96]) * se / pred)
            if np.isfinite(hi):
                return lo, hi
            return max(pred - 1.96 * se, 0.0), pred + 1.96 * se
        if link == 'logit' and 0 < pred < 1:
            lo, hi = expit(np.log(pred / (1 - pred)) + np.array([-1.96, 1.96]) * se / (pred * (1 - pred)))
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi
            return max(pred - 1.96 * se, 0.0), min(pred + 1.96 * se, 1.0)
    return pred - 1.96 * se, pred + 1.96 * se


def _estimate_column(fit, rows, cells, estimands, eval_at):
    """Rows of the output frame for one table column: `cells` is [(assign, meta)] and
    `estimands` is [(weights over cells, meta)] -- cells themselves are unit weights."""
    stats = []
    for assign, _ in cells:
        X = fit.spec.design_matrix(rows, assign)
        if eval_at == 'mean':
            X = X.mean(axis=0, keepdims=True)
        elif eval_at == 'median':
            X = np.median(X, axis=0, keepdims=True)
        stats.append(_cell_stats(X, fit))

    W = np.array([w for w, _ in estimands])
    m = len(estimands)
    se, p, lo, hi = (np.full(m, np.nan) for _ in range(4))

    # same spurious BLAS warnings as in _cell_stats (W is mostly zeros)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        est = W @ np.array([s[0] for s in stats])
        if fit.boot is not None:
            D = np.column_stack([s[2] for s in stats]) @ W.T          # (draws, m)
            se = D.std(axis=0)
            p = 2 * np.minimum((D > 0).mean(axis=0), (D < 0).mean(axis=0))
            lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
        elif fit.cov is not None:
            G = W @ np.array([s[1] for s in stats])                   # (m, k)
            se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', G, fit.cov, G), 0.0))
            p = 2 * (1 - norm.cdf(np.abs(est / se)))
            lo, hi = est - 1.96 * se, est + 1.96 * se

    out = []
    for i, (_, meta) in enumerate(estimands):
        row = dict(meta, estimate=est[i], se=se[i], p=p[i], ci_lo=lo[i], ci_hi=hi[i], n=len(rows))
        if meta['kind'] in ('cell', 'reference'):
            row['p'] = np.nan  # testing a predicted level against 0 isn't meaningful
            if fit.cov is not None and fit.boot is None:
                row['ci_lo'], row['ci_hi'] = _cell_ci(est[i], se[i], fit.link)
        out.append(row)
    return out


def predicted_outcomes(fit, df, comparison, columns=None, hold=None, eval_at='ame',
                       reference_row=False, verbose=True):
    """
    Predicted outcome for each of the comparison's 2 x 2 cells, the effect at each level
    of `across`, and the difference between those effects -- with SEs from fit.boot
    (bootstrap) if present, else fit.cov (delta method), else point estimates only.

    columns   None for a single 'Estimate' column, a Sweep to repeat everything with a
              variable set to each of several levels, or a Strata to repeat it within
              bins of the sample.
    hold      {Var: level} to set for every prediction (a number or Q); every other Var
              not set by the comparison/columns keeps each row's own value.
    eval_at   'ame'           predict each row of df under the cell's assignment, keeping
                              its other covariates, and average (never extrapolates past
                              covariate combinations actually observed; the right default
                              under a nonlinear link).
              'mean'/'median' MEM: predict once at the column-wise mean/median design row.
                              Equal to 'ame' under an identity link; under log/logit it can
                              land far from any real observation.
    reference_row  add a 'Sample Average' row: the average prediction with only the
              columns/hold assignments applied (nobody's effect/across value forced) --
              a baseline for reading the cells.
    """
    assert eval_at in ('ame', 'mean', 'median')
    spec, cmp = fit.spec, comparison
    _check_in_spec(spec, cmp.effect, 'effect')
    _check_in_spec(spec, cmp.across, 'across')

    base = {v.var: resolve_level(val, df, v.var) for v, val in (hold or {}).items()}
    effect = [(name, resolve_level(v, df, cmp.effect.var)) for name, v in cmp.effect_levels.items()]
    across = [(name, resolve_level(v, df, cmp.across.var)) for name, v in cmp.across_levels.items()]

    # (column name, rows, extra assignment) for each table column
    if columns is None:
        col_specs, title = [('Estimate', df, {})], None
    elif isinstance(columns, Sweep):
        _check_in_spec(spec, columns.var, 'sweep variable')
        col_specs = [(name, df, {columns.var.var: resolve_level(v, df, columns.var.var)})
                     for name, v in columns.levels.items()]
        title = columns.title or columns.var.label
    elif isinstance(columns, Strata):
        s = df[columns.var.var]
        bin_id = (pd.qcut(s, columns.bins, labels=columns.labels) if isinstance(columns.bins, int)
                  else pd.cut(s, columns.bins, labels=columns.labels))
        col_specs = [(str(b), df[bin_id == b], {}) for b in bin_id.cat.categories if (bin_id == b).any()]
        title = columns.title or columns.var.label
    else:
        raise TypeError("columns must be None, a Sweep, or a Strata")

    # cells and the estimands built from them (weights over cells), in display order
    (e0, _), (e1, _) = effect
    (a0, _), (a1, _) = across
    cell_keys = [(a, e) for a, _ in across for e, _ in effect]
    n_cells = len(cell_keys) + reference_row

    def unit(i):
        w = np.zeros(n_cells)
        w[i] = 1.0
        return w

    estimands = [(unit(i), {'kind': 'cell', 'across': a, 'effect': e, 'label': f'{a} {e}'})
                 for i, (a, e) in enumerate(cell_keys)]
    if reference_row:
        estimands.append((unit(len(cell_keys)), {'kind': 'reference', 'across': None,
                                                 'effect': None, 'label': 'Sample Average'}))
    effect_w = {}
    for a, _ in across:
        effect_w[a] = unit(cell_keys.index((a, e0))) - unit(cell_keys.index((a, e1)))
        estimands.append((effect_w[a], {'kind': 'effect', 'across': a, 'effect': None,
                                        'label': f'{a} {cmp.effect_name}'}))
    estimands.append((effect_w[a1] - effect_w[a0], {'kind': 'difference', 'across': None,
                                                    'effect': None, 'label': cmp.difference_name}))

    records = []
    for col_name, rows, col_assign in col_specs:
        assign0 = {**base, **col_assign}
        cells = [({**assign0, cmp.across.var: av, cmp.effect.var: ev}, (a, e))
                 for a, av in across for e, ev in effect]
        if reference_row:
            cells.append((assign0, None))
        for r in _estimate_column(fit, rows, cells, estimands, eval_at):
            records.append({'column': col_name, **r})

    result = PredictedOutcomes(pd.DataFrame.from_records(records), cmp,
                               [c for c, _, _ in col_specs], title, eval_at, fit.link)
    if verbose:
        print_predicted_outcomes(result)
    return result


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _stars(p):
    if p is None or np.isnan(p):
        return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''


def print_predicted_outcomes(result):
    eval_desc = {'mean': 'mean', 'median': 'median', 'ame': "each observation's own values (averaged)"}
    for col in result.columns:
        f = result.frame[result.frame['column'] == col]
        head = 'PREDICTED OUTCOMES' + (f' | {result.column_title} = {col}' if result.column_title else '')
        print('\n' + '=' * 78 + f'\n{head} (n={f["n"].iloc[0]})')
        print(f'(Other variables held at {eval_desc[result.eval_at]})\n' + '=' * 78)
        print(f"{'':40} {'Estimate':>10} {'SE':>8} {'95% CI':>20} {'p':>7}")
        prev_kind = None
        for _, r in f.iterrows():
            if r['kind'] == 'effect' and prev_kind != 'effect':
                print('-' * 78)
            prev_kind = r['kind']
            if np.isnan(r['se']):
                print(f"{r['label']:40} {r['estimate']:10.4f}")
                continue
            p = '' if np.isnan(r['p']) else f"{r['p']:7.3f}{_stars(r['p'])}"
            print(f"{r['label']:40} {r['estimate']:10.4f} {r['se']:8.4f} "
                  f"[{r['ci_lo']:8.4f}, {r['ci_hi']:8.4f}] {p}")


def export_predicted_outcomes_table(result, caption, label, widthmultiplier=1, notes=None,
                                     column_group=None):
    """Write a PredictedOutcomes as a LaTeX table to tables/<label suffix>.tex.

    Layout: Panel A reports the predicted rate for each cell, grouped under the two
    `across` levels; Panel B the effect (first effect level minus second) at each `across`
    level and, in bold, their difference. SEs sit on their own line below each estimate,
    and significance stars are \\rlap'd so decimal points stay aligned.

    column_group is the spanning header over the columns (default: the Sweep/Strata title;
    ignored for a single column). widthmultiplier=None keeps the table at its natural
    width; a number stretches it to that fraction of \\textwidth. Needs booktabs +
    threeparttable."""
    cmp = result.comparison
    f = result.frame
    lookup = {(r['column'], r['label']): r for _, r in f.iterrows()}
    cols = result.columns
    ncol = len(cols)
    is_multi = result.column_title is not None
    column_group = column_group or result.column_title

    headers = list(cols) if is_multi else ['Estimate']
    # '10th Percentile', ... -> '10th', ... under a '<column_group> (Percentile)' header
    if is_multi and column_group and all(h.endswith(' Percentile') for h in headers):
        headers = [h[:-len(' Percentile')] for h in headers]
        column_group = f'{column_group} (Percentile)'

    def num(x, bold=False):
        s = f'{x:.3f}'.replace('-', '$-$')
        return f'\\textbf{{{s}}}' if bold else s

    def stars(p):
        s = _stars(p)
        return f'\\rlap{{$^{{{s}}}$}}' if s else ''

    def rows(row_label, key, bold=False, indent=True, sublabel=''):
        """Estimate line plus (when there are SEs) a parenthesized SE line; `sublabel`
        goes in the SE line's label cell, to split a long label over both lines."""
        vals = [lookup[(c, key)] for c in cols]
        pad = '\\quad ' if indent else ''
        lbl = f'\\textbf{{{row_label}}}' if bold else row_label
        out = [f'{pad}{lbl} & ' + ' & '.join(num(v['estimate'], bold) + stars(v['p']) for v in vals) + ' \\\\']
        if any(not np.isnan(v['se']) for v in vals):
            ses = ' & '.join('' if np.isnan(v['se']) else f"({v['se']:.3f})" for v in vals)
            out.append(f'{sublabel} & {ses} \\\\[4pt]')
        return out

    # panel titles sit in the label column (not a full-width \multicolumn, which would
    # dump any extra width into the last numeric column of a narrow table)
    def panel(title):
        return [f'\\textit{{{title}}}' + ' &' * ncol + ' \\\\', '\\addlinespace[2pt]']

    across, effect = list(cmp.across_levels), list(cmp.effect_levels)
    body = panel('Panel A: Predicted Probability')
    for a in across:
        body.append(a + ' &' * ncol + ' \\\\')
        for e in effect:
            body += rows(e, f'{a} {e}')
    if (f['kind'] == 'reference').any():
        body += rows('Sample Average', 'Sample Average', indent=False)
    body += ['\\midrule'] + panel(f'Panel B: {cmp.effect_name}')
    for a in across:
        body += rows(a, f'{a} {cmp.effect_name}', indent=False)
    body.append('\\addlinespace[2pt]')
    body += rows(cmp.difference_name, cmp.difference_name, bold=True, indent=False,
                 sublabel=f'({across[1]} $-$ {across[0]})')
    body[-1] = body[-1].replace('\\\\[4pt]', '\\\\')

    head = []
    if is_multi and column_group:
        head += [f' & \\multicolumn{{{ncol}}}{{c}}{{{column_group}}} \\\\',
                 f'\\cmidrule(l){{2-{ncol + 1}}}']
    head.append(' & ' + ' & '.join(headers) + ' \\\\')

    # extra right padding on each numeric column leaves room for the \rlap'd stars
    colspec = 'l' + 'r@{\\hspace{1.2em}}' * ncol
    if widthmultiplier is None:
        begin, end = f'\\begin{{tabular}}{{{colspec}}}', '\\end{tabular}'
    else:
        begin = f'\\begin{{tabular*}}{{{widthmultiplier}\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}'
        end = '\\end{tabular*}'

    notes_block = []
    if notes:
        items = [f'\\item {n}' for n in ([notes] if isinstance(notes, str) else notes)]
        notes_block = ['\\begin{tablenotes}[flushleft]', '\\footnotesize', *items, '\\end{tablenotes}']

    text = '\n'.join([
        '\\begin{table}[h]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        '\\begin{threeparttable}', begin, '\\toprule', *head, '\\midrule', *body,
        '\\bottomrule', end, *notes_block, '\\end{threeparttable}', '\\end{table}',
    ]) + '\n'
    with open('tables/' + label.split(':')[-1] + '.tex', 'w') as fh:
        fh.write(text)
