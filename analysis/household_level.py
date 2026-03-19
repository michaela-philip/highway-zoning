import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sample = pd.read_pickle('data/output/household_sample.pkl')
zoning_map = {'dwelling': 'residential', 'apartment': 'residential', 
                    'single family': 'residential', '2 family': 'residential', 
                    '2-4 family': 'residential', 'commercial': 'industrial',
                    'industrial': 'industrial', 'business': 'industrial',
                    'light industry': 'industrial', 'heavy industry': 'industrial'
                    }

sample['zoning'] = sample['Zonetype'].map(zoning_map)
sample['Residential'] = np.where(sample['zoning']=='residential', 1, 0)

model_1945def = 'hwy ~ black + Residential + (black * Residential) + C(city)'
results = smf.ols(model_1945def, data=sample).fit(cov_type='cluster', cov_kwds={'groups': sample['city']})
print(results.summary())