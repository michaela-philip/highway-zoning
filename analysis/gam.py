import pandas as pd
import numpy as np
from pygam import LogisticGAM, s, l, f, te
from pygam.terms import TermList
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import reduce
from operator import add

df = pd.read_pickle('data/output/sample.pkl')

# ── 1. PREP ──────────────────────────────────────────────────────────────────
# Transformations suggested by the GAM
df['log_dist_to_hwy'] = np.log1p(df['dist_to_hwy'])
df['log_dist_to_rr']  = np.log1p(df['dist_to_rr'])
df['log_rent']        = np.log1p(df['rent'])
df['log_valueh']      = np.log1p(df['valueh'])
df['distance_to_cbd_sq'] = df['distance_to_cbd'] ** 2
df['dm_elevation_sq']    = df['dm_elevation'] ** 2
df['dist_water_sq']      = df['dist_water'] ** 2

# Your treatment variables (kept linear — these are your B_1, B_2, B_3)
linear_vars = ['mblack_mean_pct', 'Residential', 'black_x_residential']
df['black_x_residential'] = df['mblack_mean_pct'] * df['Residential']

# Controls to model flexibly
smooth_vars = [
    'distance_to_cbd_sq', 'dist_water_sq', 'log_rent', 'log_valueh',
    'owner', 'dm_elevation_sq', 'slope', 'log_dist_to_rr', 'log_dist_to_hwy', 'numprec'
]

# flood_risk is binary — enters as a linear/factor term, not a spline
binary_controls = ['flood_risk']

all_vars = linear_vars + smooth_vars + binary_controls

df_model = df[['hwy'] + all_vars + ['city']].dropna()

# Standardize smooth controls (helps spline fitting)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_model[smooth_vars] = scaler.fit_transform(df_model[smooth_vars])

X = df_model[all_vars].values
y = df_model['hwy'].values

# Column indices
linear_idx  = [all_vars.index(v) for v in linear_vars]
smooth_idx  = [all_vars.index(v) for v in smooth_vars]
binary_idx  = [all_vars.index(v) for v in binary_controls]

# ── 2. BUILD TERM LIST ────────────────────────────────────────────────────────
def build_terms(term_list):
    return reduce(add, term_list)

terms = build_terms(
    [l(i) for i in linear_idx]
  + [s(i, n_splines=10) for i in smooth_idx]
  + [l(i) for i in binary_idx]
)

# ── 3. FIT ────────────────────────────────────────────────────────────────────

# LogisticGAM because hwy is a 0/1 dummy (linear probability via GAM(distribution='binomial'))
gam = LogisticGAM(terms)
gam.gridsearch(X, y)        # tunes the smoothing penalty lambda via GCV
print(gam.summary())

# ── 4. EXTRACT TREATMENT COEFFICIENTS ────────────────────────────────────────

# These are your B_1, B_2, B_3 analogues
summary = gam.summary()

# Confidence intervals on linear terms
for i, name in zip(linear_idx, linear_vars):
    ci = gam.confidence_intervals(X, width=0.95)
    print(f"{name}: coef = {gam.coef_[i]:.4f}")

# ── 5. PLOT SMOOTH TERMS ──────────────────────────────────────────────────────
# This is the main diagnostic — look for nonlinearities in your controls

n_smooth = len(smooth_vars)
ncols = 3
nrows = int(np.ceil(n_smooth / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
axes = axes.flatten()

for plot_i, (var, col_i) in enumerate(zip(smooth_vars, smooth_idx)):
    ax = axes[plot_i]

    # Generate grid over this term holding others at mean
    XX = gam.generate_X_grid(term=col_i)
    pdep, confi = gam.partial_dependence(term=col_i, X=XX, width=0.95)

    ax.plot(XX[:, col_i], pdep, color='steelblue', lw=2)
    ax.fill_between(XX[:, col_i], confi[:, 0], confi[:, 1],
                    alpha=0.3, color='steelblue')
    ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    ax.set_title(var, fontsize=11)
    ax.set_xlabel('Standardized value')
    ax.set_ylabel('Partial effect (log-odds)')

# Hide unused subplots
for ax in axes[n_smooth:]:
    ax.set_visible(False)

plt.suptitle('GAM Smooth Terms — Control Variables\n(log-odds of highway placement)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('tables/gam_smooth_terms.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 6. CITY FIXED EFFECTS (optional but recommended) ─────────────────────────
# If you want city FEs, easiest to dummy them out and add as linear terms
# since pygam doesn't have a native FE syntax

city_dummies = pd.get_dummies(df_model['city'], drop_first=True, dtype=float)
city_cols = city_dummies.columns.tolist()
df_model = pd.concat([df_model, city_dummies], axis=1)

city_idx = [len(all_vars) + i for i in range(len(city_cols))]
all_vars_fe = all_vars + city_cols
X_fe = df_model[all_vars_fe].values

terms_fe = build_terms(
    [l(i) for i in linear_idx]
  + [s(i, n_splines=10) for i in smooth_idx]
  + [l(i) for i in binary_idx]
  + [l(i) for i in city_idx]
)
gam_fe = LogisticGAM(terms_fe)
gam_fe.gridsearch(X_fe, y)
print(gam_fe.summary())