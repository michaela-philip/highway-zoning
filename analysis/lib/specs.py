import statsmodels.api as sm

# Each spec is a list of (variable_name, display_label) pairs, kept together so the
# two can't drift out of alignment the way separately-maintained x_vars/columns lists did.

CORE_VARS = [
    ('Residential', 'Residential'),
    ('mblack_1945def', 'Black'),
    ('ResidentialxBlack', 'Residential x Black'),
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
    ('dm_elevation', 'Elevation'),
]

HH_CONTROLS = [
    ('owner', 'Owner'),
    ('numprec', 'Number of Residents'),
]

LOG_DIST_HWY = [
    ('log_dist_to_hwy', 'Log(Distance to Highway)'),
]

CNN_PROB = [
    ('prob_hwy', 'Probability of Highway (CNN)'),
]

CNN_INTERACTIONS = [
    ('BlackxProbHwy', 'Black x Probability of Highway'),
    ('ResidentialxProbHwy', 'Residential x Probability of Highway'),
    ('ResidentialxBlackxProbHwy', 'Residential x Black x Probability of Highway'),
]

DEM_ACCESS = [
    ('dem_access_norm', 'Demographic Access (Normalized)'),
    ('ResidentialxAccess', 'Residential x Demographic Access'),
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


def fit_ols(df, x_vars, columns, cluster_var='city'):
    """OLS of 'hwy' on x_vars, clustered by cluster_var, with the design matrix
    relabeled to the friendly `columns` so results.params is indexed the same way
    bootstrap_lpm_table results are (columns[0] is 'Intercept', matching sm.add_constant)."""
    X = df[x_vars].copy()
    X.columns = columns[1:]
    X = sm.add_constant(X)
    return sm.OLS(df['hwy'], X).fit(cov_type='cluster', cov_kwds={'groups': df[cluster_var]})
