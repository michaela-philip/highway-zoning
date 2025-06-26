import pandas as pd

census = '/homes/nber/philipm/bulk/jones_ipums549/census-ipums/current/csv/1940.csv'

#chunk_counter = 0
# remember that this appends - if you need to rerun this code delete the existing csv first
#for chunk in pd.read_csv(census, chunksize=1000000, low_memory=False):
    #chunk = chunk[chunk['statefip'] == 13]
    
    #if not chunk.empty:  
        #chunk.to_csv('data/output/census_ga_1940.csv', mode='a', header=(chunk_counter == 0), index=False)
        #chunk_counter += 1
        #print(f'Chunk {chunk_counter} processed')

#print("Processing complete!")

chunks = []
for chunk in pd.read_csv(census, chunksize=1000000, low_memory=False):
    chunk = chunk[chunk['statefip'] == 13]
    if not chunk.empty:
        chunks.append(chunk)
        print(f'Chunk with {len(chunk)} rows processed')

# Concatenate all chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)
df.to_pickle('data/input/ga_1940.pkl')
print("Processing complete and pickled!")