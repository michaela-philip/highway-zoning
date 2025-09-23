import pandas as pd

census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/current/csv/1940.csv'

def get_census(states):
    chunks = []
    for chunk in pd.read_csv(census, chunksize=1000000, low_memory = False):
        chunk = chunk[chunk['statefip'].isin(states)]
        if not chunk.empty:
            chunks.append(chunk)
            print(f'Chunk with {len(chunk)} rows processed')

    df = pd.concat(chunks, ignore_index=True)
    df.to_pickle('data/input/census_1940.pkl')
    print("Processing complete and pickled!")

states = [13, 21] # Georgia and Kentucky FIPS
get_census(states)
