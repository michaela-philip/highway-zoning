# A regression spec is built from term objects that know how to compute their own column
# from the data -- not from (column name, label) pairs pointing at columns precomputed on
# df. That's what keeps fitting and counterfactual prediction in sync: both call
# Spec.design_matrix, and a prediction just tells it which base variables to override
# (`assign`). Every interaction / spline / indicator built on top of an overridden variable
# is recomputed from it automatically, so there's no separate "which columns depend on
# what" bookkeeping to fall out of date.
#
#   Var          a regressor read straight from a df column (the only thing `assign` sets)
#   Transform    a function of one Var (e.g. Var.hinge for a linear-spline term)
#   Interaction  a product of Vars/Transforms (build with `a * b`, interact(), cross())
#   Spec         an ordered, de-duplicated list of the above -> design matrix + labels
#
# Labels are the real display labels used in every exported table ('Exposure to Black
# Population x High Suitability', ...), generated from the component labels, so there's
# nothing to rename afterwards.

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Q:
    """A data-dependent level: the q-th quantile of whatever Var it's attached to,
    resolved on the sample being predicted on (optionally restricted to rows where
    where(df) is True -- e.g. where=lambda d: d['hwy'] == 1)."""
    q: float
    where: Callable = None

    def resolve(self, df, var):
        s = df[var]
        if self.where is not None:
            s = s[self.where(df)]
        return float(s.quantile(self.q))


def resolve_level(value, df, var):
    """A level's numeric value: a plain number as-is, a Q resolved on df[var]."""
    return value.resolve(df, var) if isinstance(value, Q) else value


def _ordinal(n):
    n = int(n)
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def percentile_levels(percentiles, where=None):
    """{'10th Percentile': Q(0.10, where), ...} for the given percentiles (0-100) -- e.g.
    the levels of a Sweep over suitability among squares that got a highway."""
    return {f'{_ordinal(p)} Percentile': Q(p / 100, where) for p in percentiles}


class _Term:
    """Shared behavior for anything that can be a regressor."""
    label: str
    key: tuple

    def values(self, df, assign):
        raise NotImplementedError

    def parts(self):
        return (self,)

    def __mul__(self, other):
        return Interaction(self, other)

    def __repr__(self):
        return f'{type(self).__name__}({self.label!r})'


class Var(_Term):
    """A regressor read straight from df[var].

    levels: optional default {name: value} evaluation levels, in order, for when this Var
    plays a role in a Comparison (e.g. {'Non-Residential': 0, 'Residential': 1}, or
    {'Low Exposure': Q(.25), 'High Exposure': Q(.75)} for a continuous measure).
    note: optional prose definition, for table notes."""

    def __init__(self, var, label, levels=None, note=None):
        self.var, self.label, self.levels, self.note = var, label, levels, note
        self.key = ('var', var)

    def values(self, df, assign):
        if self.var in assign:
            return np.broadcast_to(np.asarray(assign[self.var], dtype=float), (len(df),))
        return df[self.var].to_numpy(dtype=float)

    def transform(self, fn, label, key=None):
        """A Transform computing fn(this Var's values). `key` identifies the function for
        de-duplication; defaults to the label."""
        return Transform(self, fn, label, key or label)

    def hinge(self, knot, label):
        """max(0, x - knot): the above-knot piece of a linear spline in this Var."""
        return Transform(self, lambda x: np.maximum(0.0, x - knot), label, ('hinge', float(knot)))

    def bins(self, cutpoints, labels):
        """Indicators for this Var falling in each bin above the lowest, where `cutpoints`
        are the interior bin edges (k cutpoints -> k+1 bins -> k indicators, one per
        label; the lowest bin is the omitted baseline). E.g. quartile dummies:
        PCT_RES.bins(df['pct_res'].quantile([.25, .5, .75]).tolist(), ['Q2', 'Q3', 'Q4'])."""
        cutpoints = [float(c) for c in cutpoints]
        assert len(labels) == len(cutpoints), "need one label per cutpoint (= per non-baseline bin)"
        edges = cutpoints + [np.inf]
        return [Transform(self, lambda x, lo=lo, hi=hi: ((x >= lo) & (x < hi)).astype(float),
                          f'{self.label}: {lab}', ('bin', lo, hi))
                for lo, hi, lab in zip(edges[:-1], edges[1:], labels)]


class Transform(_Term):
    """fn(parent's values), recomputed from the parent whenever the parent is assigned."""

    def __init__(self, parent, fn, label, fn_key):
        self.parent, self.fn, self.label = parent, fn, label
        self.key = ('tf', parent.key, fn_key)

    def values(self, df, assign):
        return self.fn(self.parent.values(df, assign))


class Interaction(_Term):
    """Product of Vars/Transforms. Nested products flatten, so (a * b) * c == a * (b * c);
    the label keeps construction order, the key ignores it (a * b de-duplicates with b * a)."""

    def __init__(self, *terms):
        self._parts = tuple(p for t in terms for p in t.parts())
        assert len(self._parts) >= 2
        self.label = ' x '.join(p.label for p in self._parts)
        self.key = ('x', frozenset(p.key for p in self._parts))

    def parts(self):
        return self._parts

    def values(self, df, assign):
        out = self._parts[0].values(df, assign)
        for p in self._parts[1:]:
            out = out * p.values(df, assign)
        return out


def _as_group(g):
    return list(g) if isinstance(g, (list, tuple)) else [g]


def interact(*groups, min_order=1):
    """Full factorial of the given terms: every main effect (unless min_order > 1) and
    every product of 2, 3, ... of them. A group passed as a list (e.g. the output of
    Var.bins) is treated as one categorical: each of its terms enters separately and
    terms from the same group are never multiplied together.

      interact(RES, BLACK)        -> RES, BLACK, RES x BLACK
      interact(RES, BLACK, SUIT)  -> + SUIT, RES x SUIT, BLACK x SUIT, RES x BLACK x SUIT
    """
    groups = [_as_group(g) for g in groups]
    out = []
    for order in range(min_order, len(groups) + 1):
        for subset in combinations(groups, order):
            for combo in product(*subset):
                out.append(combo[0] if order == 1 else Interaction(*combo))
    return out


def cross(block, *others):
    """Each term in `block` multiplied by `others` -- e.g. cross(interact(RES, BLACK),
    HIGH_SUIT) gives RES x HIGH_SUIT, BLACK x HIGH_SUIT, RES x BLACK x HIGH_SUIT. A list
    in `others` is a categorical group, as in interact()."""
    out = []
    for t in block:
        for combo in product(*(_as_group(o) for o in others)):
            out.append(Interaction(t, *combo))
    return out


def _flatten(blocks):
    for b in blocks:
        if isinstance(b, _Term):
            yield b
        else:
            yield from _flatten(b)


def labels(*blocks):
    """Display labels of every term in the given blocks -- e.g. the `keep` list for
    leaveout_except."""
    return [t.label for t in _flatten(blocks)]


class Spec:
    """An ordered set of regressors (plus intercept). Blocks may overlap -- a term that
    appears twice (same key) is kept once, in its first position. Two different terms
    with the same label, or the same term under two different labels, is an error."""

    def __init__(self, *blocks):
        terms, by_key, by_label = [], {}, {}
        for t in _flatten(blocks):
            if t.key in by_key:
                if by_key[t.key].label != t.label:
                    raise ValueError(f"{t.key} appears with two labels: "
                                     f"{by_key[t.key].label!r} and {t.label!r}")
                continue
            if t.label in by_label:
                raise ValueError(f"two different terms share the label {t.label!r}")
            by_key[t.key] = by_label[t.label] = t
            terms.append(t)
        self.terms = terms

    @property
    def columns(self):
        return ['Intercept'] + [t.label for t in self.terms]

    def __contains__(self, term):
        return any(t.key == term.key for t in self.terms)

    def design_matrix(self, df, assign=None):
        """(n, k) float array [intercept, *terms] -- for fitting (assign=None) or for a
        counterfactual where assign={var name: value} overrides those base columns
        (scalar or length-n array) before every term is computed."""
        assign = assign or {}
        X = np.empty((len(df), len(self.terms) + 1))
        X[:, 0] = 1.0
        for j, t in enumerate(self.terms, start=1):
            X[:, j] = t.values(df, assign)
        return X

    def __repr__(self):
        return 'Spec(\n  ' + '\n  '.join(self.columns) + '\n)'


def leaveout_except(columns, keep):
    """Labels to drop from an exported table: everything except `keep`."""
    return [c for c in columns if c not in keep]


# --------------------------------------------------------------------------
# variable definitions
# --------------------------------------------------------------------------

BINARY_BLACK_LEVELS = {'White': 0, 'Black': 1}
EXPOSURE_LEVELS = {'Low Exposure': Q(0.25), 'High Exposure': Q(0.75)}

RES = Var('Residential', 'Residential', levels={'Non-Residential': 0, 'Residential': 1},
          note='Residential is a binary variable indicating whether 75\\% or more of the square is zoned for residential use.')
PCT_RES = Var('pct_res', 'Percent Residential')

# Alternative definitions of "Black"
MBLACK_60 = Var('mblack_1945def', 'Majority Black', BINARY_BLACK_LEVELS,
                'majority-Black, defined as greater than 60 percent Black (the 1945 legal definition)')
MBLACK_50 = Var('mblack_50_pct', 'Majority Black', BINARY_BLACK_LEVELS,
                'majority-Black, defined as greater than 50 percent Black')
MBLACK_40 = Var('mblack_40_pct', 'Majority Black', BINARY_BLACK_LEVELS,
                'majority-Black, defined as greater than 40 percent Black')
MBLACK_MEAN_PCT = Var('mblack_mean_pct', 'Black', BINARY_BLACK_LEVELS,
                      "an indicator for a square's percent Black being above its city's mean")
MBLACK_MEAN_SHARE = Var('mblack_mean_share', 'Black', BINARY_BLACK_LEVELS,
                        "an indicator for a square's share of its city's Black population being above the city's mean share")
LOG_BLACK = Var('log_black', 'Log(Percent Black)', note='the log percent Black population')
ANY_BLACK = Var('any_black', 'Any Black Residents', {'No Black Residents': 0, 'Black Residents': 1},
                'the presence of any Black residents')
MIXED_BLACK = Var('mixed_black', 'Mixed Black', BINARY_BLACK_LEVELS,
                  'mixed-Black, defined as between 30 and 80 percent Black')
HIGH_BLACK_SHARE = Var('high_black_share', 'High Black Share', BINARY_BLACK_LEVELS,
                       'indicator for having a high share of Black residents')
DEM_ACCESS = Var('dem_access', 'Exposure to Black Population', EXPOSURE_LEVELS,
                 'distance-decayed demographic access to the Black population')
HIGH_DEM_ACCESS = Var('high_dem_access', 'High Exposure to Black Population', BINARY_BLACK_LEVELS,
                      'indicator for high access to the Black population')
BLACK_HOMEOWNERS_ACCESS = Var('black_homeowners_access', 'Exposure to Black Homeowners', EXPOSURE_LEVELS,
                              'distance-decayed access to Black homeowners')
ANY_BLACK_HOMEOWNERS = Var('black_homeowners_indicator', 'Any Black Homeowners',
                           {'No Black Homeowners': 0, 'Black Homeowners': 1},
                           'indicator for presence of Black homeowners')
HOMEOWNERS_ACCESS = Var('homeowners_access', 'Exposure to Homeowners', EXPOSURE_LEVELS,
                        'distance-decayed access to homeowners')

# CNN highway suitability
SUIT = Var('logit_normalized', 'Est. Highway Suitability')
CNN_LOGIT = [SUIT]
CNN_PROB = [Var('prob_hwy', 'Probability of Highway (CNN)')]

# Control blocks
CONTINUOUS_EXPOSURE = [
    Var('res_access', 'Residential Access'),
    Var('dem_access', 'Demographic Access'),
    Var('resxdem_access', 'Residential x Demographic Access'),
]

BLACK_HOMEOWNERSHIP_EXPOSURE = [Var('black_homeowners_access', 'Black Homeowners Access')]

RESIDENTIAL = [RES]

HOUSING_VARS = [
    Var('log_valueh', 'Log(Value)'),
    Var('log_rent', 'Log(Rent)'),
]

GEO_CONTROLS = [
    Var('log_dist_to_rr', 'dist(RR)'),
    Var('log_dist_to_rr_sq', 'dist(RR^2)'),
    Var('distance_to_cbd', 'dist(CBD)'),
    Var('distance_to_cbd_sq', 'dist(CBD^2)'),
    Var('flood_risk', 'Flood Risk'),
    Var('dist_water', 'dist(Water)'),
    Var('slope', 'Slope'),
    Var('elevation', 'Elevation'),
]

HWY_40 = [Var('hwy_40', 'Intersected by 1940 Highway')]

HH_CONTROLS = [
    Var('owner', 'Percent Owner-Occupied'),
    Var('log_numprec', 'Log(Number of Residents)'),
]

HH_RACE_CONTROLS = HH_CONTROLS + [Var('log_black_residents', 'Log(Black Residents)')]

LOG_DIST_HWY = [
    Var('dist_to_hwy', 'Distance to Highway'),
    Var('dist_to_hwy_sq', 'Distance to Highway^2'),
]

HWY_ACCESS = [
    Var('hwy_access', 'Distance-Decayed Highway Access'),
    Var('hwy_access_wide', 'Distance-Decayed Highway Access (wide)'),
]

CITY_LABELS = {'louisville': 'City_Louisville', 'littlerock': 'City_LittleRock'}


def city_dummies(df):
    """Dummies for every city in df but the first (the omitted baseline)."""
    cities = list(df['city'].unique())
    return [Var(f'city_{c}', CITY_LABELS.get(c, f'City_{c.title()}')) for c in cities[1:]]
