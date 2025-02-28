import pandas as pd

ga = pd.read_csv('data/output/census_ga.csv')

# drop all rural observations 
ga = ga[ga['urban'] == 2]

