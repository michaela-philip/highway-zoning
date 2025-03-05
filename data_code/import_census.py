import pandas as pd

census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/v2024/csv/1950.csv'

chunk_counter = 0
# remember that this appends - if you need to rerun this code delete the existing csv first
for chunk in pd.read_csv(census, chunksize=1000000, low_memory=False):
    chunk = chunk[chunk['statefip'] == 13]
    
    if not chunk.empty:  
        chunk.to_csv('data/output/census_ga.csv', mode='a', header=(chunk_counter == 0), index=False)
        chunk_counter += 1
        print(f'Chunk {chunk_counter} processed')

print("Processing complete!")