import pandas as pd
import glob
import os
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd

# pull most recent activation values
dataroot = 'cnn/multiclass/'
csv_files = glob.glob(os.path.join(dataroot, '*.csv'))
csv = max(csv_files, key=os.path.getmtime)
activations = pd.read_csv(csv)
print(activations.info())

# convert x and y to real coordinates
sample = pd.read_pickle('data/output/sample.pkl')
grid_centers = (
    sample[['grid_id', 'geometry']]
    .assign(
        x_0=lambda d: d.geometry.centroid.x,
        y_0=lambda d: d.geometry.centroid.y
    )
    .set_index('grid_id')
)

activations = activations.merge(grid_centers, left_on = 's_id', right_on = 'grid_id', validate = 'many_to_one')
print(activations.info(), activations['x_0'].describe())
activations['x_abs'] = activations['x_0'] + activations['x']
activations['y_abs'] = activations['y_0'] + activations['y']
activations_gpd = gpd.GeoDataFrame(activations, geometry = gpd.points_from_xy(activations['x_abs'], activations['y_abs']), crs = sample.crs)

# spatial join to get correct grid_ids
activations_gpd = gpd.sjoin(activations_gpd, 
    sample[['grid_id', 'geometry']], how = 'left',
    predicate = 'intersects'
)
print(f'Spatial join matched {activations_gpd['grid_id'].notnull().sum()} out of {len(activations_gpd)} rows.')

# # split into real and candidate sets
# real = activations_gpd[activations_gpd['real_missing'] == True].reset_index(drop = False)
# cand = activations_gpd[activations_gpd['real_missing'] == False].reset_index(drop = False)

# # get vector of activation values
# X_real = real[real['activation']]
# X_cand = cand[cand['activation']]

# # find K nearest neighbors for each real location
# K = 2
# nn = NearestNeighbors(n_neighbors = K, metric = 'euclidean')
# nn.fit(X_cand)
# distances, indices = nn.kneighbors(X_real)

# matches = []

# # aggregate into dataframe
# for i, cand_idx_list in enumerate(indices):
#     for rank, j in enumerate(cand_idx_list):
#         matches.append({
#             "real_id": i,
#             "cand_id": j,
#             "rank": rank + 1,
#             "activation_dist": distances[i, rank]
#         })

# match_df = pd.DataFrame(matches)

# match_df = (
#     match_df
#     .merge(real, on="real_id", suffixes=("", "_real"))
#     .merge(cand, on="cand_id", suffixes=("", "_cand"))
# )

