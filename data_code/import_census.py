import pandas as pd

census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/v2024/csv/1950.csv'

chunk_counter = 0

for chunk in pd.read_csv(census, chunksize = 1000000):
    chunk = chunk[chunk['statefip'] == 13]
    chunk.to_csv('data/output/census_ga.csv', mode='a', header = (chunk_counter == 0), index = False)
    chunk_counter += 1
    print(f'Chunk {chunk_counter}')
    del chunk

print("GA complete")