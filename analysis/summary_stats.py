import pandas as pd
import numpy as np

def get_summary_stats(df, maj_black_def):
    sum_stats = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max()
    })
    

mblack_pct_ = np.where('pct_black' > )

atl_sample = pd.read_pickle('data/output/atl_sample.pkl')

sum_stats = pd.DataFrame({
    'mean': atl_sample.mean(),
    'std': atl_sample.std(),
    'min': atl_sample.min(),
    'max': atl_sample.max(),
    'observations': atl_sample.shape[0],
})

