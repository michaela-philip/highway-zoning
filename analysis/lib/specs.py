# Each spec is a list of (variable_name, display_label) pairs, kept together so the
# two can't drift out of alignment the way separately-maintained x_vars/columns lists did.

from analysis.lib.data import add_interaction

# Every regression in this project shares the same Residential / Black / Residential x
# Black core, but "Black" gets constructed differently across specs (different pct-Black
# thresholds, a continuous share, a demographic-access measure, ...). Rather than
# hand-maintaining a parallel CORE_VARS-style block (and its CNN/access interactions) per
# definition, pick a key from BLACK_DEFINITIONS and build the spec with core_spec() /
# sweep_interactions_spec() below -- the interaction columns are derived and cached on df
# on demand, and the display labels stay fixed ('Black', 'Residential x Black', ...)
# regardless of which definition backs them. Report which one was actually used (e.g. in
# a table's notes/caption) via black_definition_note().
BLACK_DEFINITIONS = {
    '60pct': ('mblack_1945def', 'majority-Black, defined as greater than 60 percent Black (the 1945 legal definition)'),
    '50pct': ('mblack_50_pct', 'majority-Black, defined as greater than 50 percent Black'),
    '40pct': ('mblack_40_pct', 'majority-Black, defined as greater than 40 percent Black'),
    'pct_mean': ('mblack_mean_pct', 'the mean percent Black population'),
    'share_mean': ('mblack_mean_share', 'the mean share of the Black population'),
    'log_pct': ('log_black', 'the log percent Black population'),
    'any': ('any_black', 'the presence of any Black residents'),
    'dem_access': ('log_dem_access', 'log distance-decayed demographic access to the Black population'),
}

RESIDENTIAL_LABEL = 'Residential'
BLACK_LABEL = 'Black'
INTERACTION_LABEL = 'Residential x Black'


def black_definition_note(black_key):
    """Human-readable description of a BLACK_DEFINITIONS key, for table notes/captions."""
    return BLACK_DEFINITIONS[black_key][1]


def core_spec(df, black_key, residential_var='Residential', black_label=BLACK_LABEL):
    """(variable, label) triple for Residential / Black / Residential x Black, where
    "Black" is whichever BLACK_DEFINITIONS[black_key] column is chosen. Labels are always
    'Residential' / black_label / 'Residential x Black' so tables read the same regardless
    of definition -- use black_definition_note(black_key) to report which one was used.
    Mutates df in place to add the interaction column if it isn't already there."""
    black_var, _ = BLACK_DEFINITIONS[black_key]
    inter_col = add_interaction(df, residential_var, black_var)
    return [
        (residential_var, RESIDENTIAL_LABEL),
        (black_var, black_label),
        (inter_col, INTERACTION_LABEL),
    ]


def sweep_interactions_spec(df, black_key, sweep_var, sweep_label, residential_var='Residential'):
    """(variable, label) triple for Residential x sweep_var, Black x sweep_var, and
    Residential x Black x sweep_var (e.g. sweep_var='logit_hwy' for the CNN-logit
    interactions) -- the sweep_interactions argument expected by
    analysis.lib.marginal_effects.predicted_outcomes/marginal_effects_table. Mirrors
    core_spec's black_key/label handling; mutates df in place as needed."""
    black_var, _ = BLACK_DEFINITIONS[black_key]
    res_sweep = add_interaction(df, residential_var, sweep_var)
    black_sweep = add_interaction(df, black_var, sweep_var)
    rb_col = add_interaction(df, residential_var, black_var)
    triple = add_interaction(df, rb_col, sweep_var)
    return [
        (res_sweep, f'{RESIDENTIAL_LABEL} x {sweep_label}'),
        (black_sweep, f'{BLACK_LABEL} x {sweep_label}'),
        (triple, f'{RESIDENTIAL_LABEL} x {BLACK_LABEL} x {sweep_label}'),
    ]


HOUSING_VARS = [
    ('log_valueh', 'Log(Value)'),
    ('log_rent', 'Log(Rent)'),
]

GEO_CONTROLS = [
    ('log_dist_to_rr', 'dist(RR)'),
    ('log_dist_to_rr_sq', 'dist(RR^2)'),
    ('distance_to_cbd', 'dist(CBD)'),
    ('distance_to_cbd_sq', 'dist(CBD^2)'),
    ('flood_risk', 'Flood Risk'),
    ('dist_water', 'dist(Water)'),
    ('slope', 'Slope'),
    ('dm_elevation', 'Elevation')
]

HH_CONTROLS = [
    ('owner', 'Percent Owner-Occupied'),
    ('numprec', 'Number of Residents'),
]

LOG_DIST_HWY = [
    ('log_dist_to_hwy', 'Log(Distance to Highway)'),
]

CNN_PROB = [
    ('prob_hwy', 'Probability of Highway (CNN)')
]

CNN_LOGIT = [
    ('logit_hwy', 'CNN Logit')
]

CITY_LABELS = {'louisville': 'City_Louisville', 'littlerock': 'City_LittleRock'}


def city_dummy_spec(df):
    """(variable, label) pairs for the non-baseline city dummies present in df."""
    cities = list(df['city'].unique())
    return [(f'city_{c}', CITY_LABELS.get(c, f'City_{c.title()}')) for c in cities[1:]]


def build_spec(df, *blocks):
    """Combine variable blocks plus city dummies into an (x_vars, columns) pair."""
    pairs = [pair for block in blocks for pair in block] + city_dummy_spec(df)
    x_vars = [v for v, _ in pairs]
    columns = ['Intercept'] + [label for _, label in pairs]
    return x_vars, columns


def leaveout_except(columns, keep):
    """Labels to drop from an exported table: everything except `keep`."""
    return [c for c in columns if c not in keep]
