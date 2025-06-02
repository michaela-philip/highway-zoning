import pandas as pd
import numpy as np
from rapidfuzz import process, distance
import os

if not os.path.exists('data/output/ga_streets.csv'):
    from scrape_streets import street_list
else:
    street_list = pd.read_csv('data/output/ga_streets.csv')

ga = pd.read_csv('data/output/census_ga_1940.csv')

# keep only columns and counties we need
cols = ['valueh', 'race', 'street', 'city', 'urban', 'countyicp', 'stateicp', 'rent', 
        'enumdist', 'respond', 'numperhh', 'numprec', 'serial', 'rawhn', 'ownershp', 'pageno']
ga2 = ga.loc[ga['countyicp'].isin([1210, 890]), cols]
atl = ga2[ga2['city'] == 350].copy()

# recode valueh and rent missing values
atl['valueh'] = atl['valueh'].replace([9999998, 9999999], np.nan)
atl['rent'] = atl['rent'].replace([0, 9998, 9999], np.nan)

# keep one observation per household/serial 
atl = atl.drop_duplicates(subset = 'serial', keep = 'first')

# function to clean and standardize addresses
def clean_addresses(df):
    # extract any additional information in () from house number
    df['street'] = df['street'].astype(str)
    df[['street', 'streetinfo']] = df['street'].str.extract(r'\s*([^\(]+)\s*(?:\(([^)]*)\))?\s*')

    # identify rawhn entries that are likely to be hotels and recode as null
    pattern = r'([0-9][ ][a-zA-Z])|([0-9][-][a-zA-Z])|([a-zA-Z][ ][0-9])|([a-zA-Z][-][0-9])'
    df.loc[df['rawhn'].str.contains(pattern, na=False), 'rawhn'] = np.nan

    # extract any non-numeric characters from rawhn and make it numeric
    df[['rawhn', 'rawhninfo']] = df['rawhn'].str.extract(r'(\d+)\s*([^\d]*)')    
    df['rawhn'] = pd.to_numeric(df['rawhn'], errors='coerce')
    
    # get previous entry's values to fill in missing streets where appropriate
    prev_rawhn = df['rawhn'].shift(1)
    prev_street = df['street'].shift(1)
    prev_page = df['pageno'].shift(1)

    ## interpolate missing address values ##
    # fill in missing street as previous street if certain conditions are met
    mask = (
        df['street'].isna() &
        df['rawhn'].notna() &
        prev_rawhn.notna() &
        (prev_page == df['pageno']) & 
        ((prev_rawhn - df['rawhn']).abs() <= 6) # norm from PVC/Logan and Zhang (2019)
    )
    df.loc[mask, 'street'] = prev_street[mask]
    print('interpolated missing street names')

    # interpolate missing house numbers for renters - use previous house number
    house_mask_rent = (
        df['rawhn'].isna() &
        df['street'].notna() & 
        df['ownershp'] != 10 &
        (prev_street == df['street'])
    )
    df.loc[house_mask_rent, 'rawhn'] = prev_rawhn[house_mask_rent]
    print('interpolated missing house numbers for renters')
    
    # interpolate missing house numbers for owners - add 2 to previous house number
    house_mask_own = (
        df['rawhn'].isna() &
        df['street'].notna() & 
        df['ownershp'] == 10 &
        (prev_street == df['street'])
    )
    df.loc[house_mask_own, 'rawhn'] = prev_rawhn[house_mask_own] + 2
    print('interpolated missing house numbers for owners')

    # additional adjustments to make matching easier
    df['street'] = (
        df['street']
        .str.lower()
        .str.replace('avenue', 'ave')
        .str.replace('street', 'st')
        .str.replace(r'( road)', 'rd', regex=True)
    )
    return df

# function to match addresses to known streets from steve morse
def match_addresses(df, streets):
    known_streets = streets['street'].unique()
    def best_match(street):
        if pd.isna(street):
            return street
        match = process.extractOne(street, known_streets,
                                            scorer = distance.JaroWinkler.distance, score_cutoff=0.2)
        return match[0] if match is not None else street
    df['street_match'] = df['street'].apply(best_match)
    return df

atl = clean_addresses(atl)
print('address cleaning done')

atl = match_addresses(atl, street_list)
print('address matching done')

atl.to_csv('data/output/atl_cleaned.csv', index=False)
print('csv created')