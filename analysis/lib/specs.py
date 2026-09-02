# Each spec is a list of (variable_name, display_label) pairs, kept together so the
# two can't drift out of alignment the way separately-maintained x_vars/columns lists did.

CORE_VARS = [
    ('Residential', 'Residential'),
    ('mblack_1945def', 'Black'),
    ('ResidentialxBlack', 'Residential x Black')
]

IMPUTED = [
    ('imputed', 'Imputed')
]

NO_RACE = [
    ('Residential', 'Residential')
]

PCT_BLACK = [
    ('Residential', 'Residential'),
    ('mblack_mean_pct', 'Black'),
    ('ResidentialxBlack_pct', 'Residential x Black')
]

SHARE_BLACK = [
    ('Residential', 'Residential'),
    ('mblack_mean_share', 'Black'),
    ('ResidentialxBlack_share', 'Residential x Black')
]

ANY_BLACK = [
    ('Residential', 'Residential'),
    ('any_black', 'Any Black Residents'),
    ('ResidentialxAnyBlack', 'Residential x Any Black')
]

RACE_OWNERSHIP = [
    ('BlackxPctOwners', 'Black x Percent Owners')
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

LOG_CNN = [
    ('log_prob', 'Log Probability of Highway (CNN)')
]

PROB_INTERACTIONS = [
    ('ResidentialxProbHwy', 'Residential x Probability of Highway'),
    ('BlackxProbHwy', 'Black x Probability of Highway'),
    ('ResidentialxBlackxProbHwy', 'Residential x Black x Probability of Highway'),
]

LOGIT_INTERACTIONS = [
    ('ResidentialxLogHwy', 'Residential x CNN Logit'),
    ('BlackxLogHwy', 'Black x CNN Logit'),
    ('ResidentialxBlackxLogHwy', 'Residential x Black x CNN Logit')
]

DEM_ACCESS = [
    ('Residential', 'Residential'),
    ('log_dem_access', 'Demographic Access'),
    ('ResidentialxAccess', 'Residential x Demographic Access'),
]

DEM_ACCESS_INTERACTIONS = [
    ('ResidentialxHwySuitability', 'Residential x Highway Suitability'),
    ('DemAccessxHwySuitability', 'Demographic Access x Highway Suitability'),
    ('ResidentialxAccessxHwySuitability', 'Residential x Demographic Access x Highway Suitability')
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
