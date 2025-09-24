import pandas as pd
import os

census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/current/csv/1940.csv'

def get_census(states):
    pickle_path = 'data/input/census_1940.pkl'
    max_state = max(states)
    if os.path.exists(pickle_path):
        print('utilizing existing census pickle')
        df = pd.read_pickle(pickle_path)
        df_states = df['stateicp'].unique().tolist()
        needed_states = [state for state in states if state not in df_states]
        if not needed_states:
            print('existing pickle contains all necessary states')
            return
        else:
            print(f'adding states {needed_states} to existing pickle')
            chunks = [] 
            for chunk in pd.read_csv(census, chunksize=1000000, low_memory = False):
                if chunk['stateicp'].min() > max_state:
                    break
                chunk = chunk[chunk['stateicp'].isin(needed_states)]
                if not chunk.empty:
                    chunks.append(chunk)
                    print(f'Chunk with {len(chunk)} rows processed')
            if chunks:
                df = pd.concat([df] + chunks, ignore_index=True)
            df.to_pickle('data/input/census_1940.pkl')
            print('processing complete and pickled!')
    else:
        print('no existing pickle, processing from csv')
        chunks = [] 
        for chunk in pd.read_csv(census, chunksize=1000000, low_memory = False):
            if chunk['stateicp'].min() > max_state:
                break
            chunk = chunk[chunk['stateicp'].isin(states)]
            if not chunk.empty:
                chunks.append(chunk)
                print(f'Chunk with {len(chunk)} rows processed')
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df.to_pickle('data/input/census_1940.pkl')
            print('processing complete and pickled!')

states = [44, 51] # Georgia and Kentucky ICPSR
get_census(states)
