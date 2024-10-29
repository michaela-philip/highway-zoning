import pandas as pd
import numpy as np
import gzip

colspecs = [(0, 4), (4, 10), (10, 18), (18, 28), (28, 30), (30, 32), (32, 36), (36, 37), (37, 38), (38, 40), (40, 44), (44, 51), (51, 60), (60, 64), (64, 74), (74, 75), (75, 78), (78, 82), (82, 88), (88, 90), (90, 125)]
columns = ['year', 'sample', 'serial', 'hhwt', 'stateicp', 'statefip', 'countyicp', 'gq', 'ownershp', 'ownershpd', 'rent', 'valueh', 'enumdist', 'pernum', 'perwt', 'race', 'raced', 'occ', 'incwage', 'versionhist', 'histid']

chunk_counter = 0

with gzip.open('data/input/usa_00005.dat.gz', 'rb') as f:
    for chunk in pd.read_fwf(f, colspecs=colspecs, header=None, chunksize=100000):
        chunk.columns = columns
        chunk = chunk[chunk['statefip'] == 13]
        chunk['ownershp'] = np.where(chunk['ownershp'] == 1, 1, 0)
        chunk.to_csv('data/output/ipums_ga.csv', mode='a', header=(chunk_counter == 0), index=False)
        chunk_counter += 1
        print(f"Chunk {chunk_counter}")
        del chunk

print("Processing complete")