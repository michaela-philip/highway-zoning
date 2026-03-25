import pandas as pd
import os

####################################################################################################
### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
states = [5, 13, 21] # Arkansas, Georgia and Kentucky FIPS

####################################################################################################
census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/1940/latest/data/csv/1940.csv'
census_othervars = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/1940/latest/data/csv/1940_othervars.csv'

def get_census(states):
    other_vars = ['histid', 'us1940b_0028']
    pickle_path = 'data/input/census_1940.pkl'
    max_state = max(states)

    if os.path.exists(pickle_path):
        print('utilizing existing census pickle')
        df = pd.read_pickle(pickle_path)
        df_states = df['statefip'].unique().tolist()
        needed_states = [state for state in states if state not in df_states]
        other_vars_needed = [var for var in other_vars if var not in df.columns]

        if not needed_states and not other_vars_needed:
            print('existing pickle contains all necessary states and variables')
            return

        if needed_states:
            print(f'adding states {needed_states} to existing pickle')
            chunks = []
            for chunk in pd.read_csv(census, chunksize=1000000, low_memory=False):
                if chunk['statefip'].min() > max_state:
                    break
                chunk = chunk[chunk['statefip'].isin(needed_states)]
                if not chunk.empty:
                    chunks.append(chunk)
                    print(f'Chunk with {len(chunk)} rows processed')
            if chunks:
                df = pd.concat([df] + chunks, ignore_index=True)

    else:
        print('no existing pickle, processing from csv')
        chunks = []
        for chunk in pd.read_csv(census, chunksize=1000000, low_memory=False):
            if chunk['statefip'].min() > max_state:
                break
            chunk = chunk[chunk['statefip'].isin(states)]
            if not chunk.empty:
                chunks.append(chunk)
                print(f'Chunk with {len(chunk)} rows processed')
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        other_vars_needed = other_vars  # always pull other vars for a fresh pickle

    if other_vars_needed:
        print(f'pulling additional variables from alt csv: {other_vars_needed}')
        usecols = list(set(['histid'] + other_vars_needed))
        histid_set = set(df['histid'])
        chunks = []
        for chunk in pd.read_csv(census_othervars, chunksize=100000, low_memory=False, usecols=usecols):
            chunk = chunk[chunk['histid'].isin(histid_set)]
            if not chunk.empty:
                chunks.append(chunk)
                print(f'Chunk with {len(chunk)} rows processed')
        if chunks:
            other_df = pd.concat(chunks, ignore_index=True)
            df = df.merge(other_df, on='histid', how='left')

    df.to_pickle(pickle_path)
    print('processing complete and pickled!')
            
get_census(states)
