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
    df['street'] = (df['street']
        .str.lower()
        .str.replace('avenue', 'ave')
        .str.replace('street', '')
        .str.replace(r'( road)', 'rd', regex=True)
        .str.replace('drive', 'dr')
        .str.replace('place', 'pl')
        .str.replace(r'( court)', 'ct', regex=True)
        .str.replace(r'( \w )', '', regex=True)
    )

    # make sure nan is correctly coded as missing
    df['street'] = df['street'].replace("nan", np.nan)
    return df

# function to match addresses to known streets from steve morse
# def match_addresses(df, streets):
    #known_streets = streets['street'].str.lower().unique()
    #prev_street = df['street'].shift(1)
    #def best_match(street):
    #    if pd.isna(street):
    #        return street
    #    match = process.extractOne(street, known_streets,
    #                                        scorer = distance.JaroWinkler.similarity, score_cutoff=0.2)
    #    return match[0] if match is not None else street
    #df['street_match'] = df['street'].apply(best_match)
    #return df

# function to match addresses to known streets from steve morse in 3 rounds
def match_addresses(df, streets):
    known_streets = streets['street'].str.lower().unique()
    df['prev_street'] = df['street'].shift(1)

    # round 1: find perfect match to known streets
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    def round_1(street):
        if pd.isna(street):
            return np.nan
        match = process.extractOne(
            street, known_streets,
            scorer=distance.JaroWinkler.normalized_similarity, 
            score_cutoff= 1.0)
        return match[0] if match is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched, 'street'].apply(round_1)

    # round 2: check if street is very similar to previous street and, if so, use that to match on
    # rationale - adjacent observations are likely to be geographically adjacent but may have typos
    mask_unmatched = df['street_match'].isna() & df['street'].notna() & df['prev_street'].notna()
    def round_2(row):
        sim = distance.JaroWinkler.normalized_similarity(row['street'], row['prev_street'])
        if sim >=0.8:
            match = process.extractOne(
            row['prev_street'], known_streets,
            scorer=distance.JaroWinkler.normalized_similarity, 
            score_cutoff = 0.2)
            return match[0] if match is not None else np.nan
        return np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_2, axis=1)
            
    # round 3: fuzzy match any remaining observations
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    def round_3(row):
            match = process.extractOne(
                row['street'], known_streets,
                scorer = distance.JaroWinkler.normalized_similarity,
                score_cutoff=0.2
            )
            return match[0] if match is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_3, axis=1)

    df.drop(columns=['prev_street'], inplace=True)
    return df


atl = clean_addresses(atl)
print('address cleaning done')

atl = match_addresses(atl, street_list)
print('address matching done')

atl.to_csv('data/output/atl_cleaned.csv', index=False)
print('csv created')