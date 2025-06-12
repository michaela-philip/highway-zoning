import pandas as pd
import numpy as np
from rapidfuzz import process, distance
import os

### FUNCTION TO CLEAN AND STANDARDIZE ADDRESSES ###
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

    ## interpolate missing address values ##
    while True:
        prev_rawhn = df['rawhn'].shift(1)
        next_rawhn = df['rawhn'].shift(-1)
        prev_street = df['street'].shift(1)
        next_street = df['street'].shift(-1)
        prev_page = df['pageno'].shift(1)
        next_page = df['pageno'].shift(-1)
        
        street_interp_forward_mask = (
            df['street'].isna() &
            df['rawhn'].notna() &
            prev_rawhn.notna() &
            prev_street.notna() &
            (prev_page == df['pageno']) & 
            ((prev_rawhn - df['rawhn']).abs() <= 6) # norm from PVC/Logan and Zhang (2019)
            )
        street_interp_back_mask = (
            df['street'].isna() &
            df['rawhn'].notna() &
            next_rawhn.notna() &
            next_street.notna() &
            (next_page == df['pageno']) & 
            ((next_rawhn - df['rawhn']).abs() <= 6) # norm from PVC/Logan and Zhang (2019)
            )

        if street_interp_back_mask.any():
            df.loc[street_interp_back_mask, 'street'] = next_street[street_interp_back_mask]
            print('interpolated missing street from following entries')
        if street_interp_forward_mask.any():
            df.loc[street_interp_forward_mask, 'street'] = prev_street[street_interp_forward_mask]
            print('interpolated missing streets from previous entries')
        if not street_interp_forward_mask.any() and not street_interp_back_mask.any():
            print('all possible streets interpolated')
            break
    
    # interpolate missing house numbers for renters - use previous house number
    print(df['rawhn'].notna().sum())
    prev_page = df['pageno'].shift(1)
    prev_street = df['street'].shift(1)
    prev_rawhn = df['rawhn'].shift(1)

    house_mask_rent = (
        df['rawhn'].isna() &
        df['street'].notna() & 
        (df['ownershp'] != 10) &
        (prev_page == df['pageno']) &
        (prev_street == df['street'])
    )
    print(prev_rawhn[house_mask_rent])
    df.loc[house_mask_rent, 'rawhn'] = prev_rawhn[house_mask_rent].values
    print('interpolated missing house numbers for renters')
    
    # interpolate missing house numbers for owners - add 2 to previous house number
    prev_page = df['pageno'].shift(1)
    prev_street = df['street'].shift(1)
    prev_rawhn = df['rawhn'].shift(1)
    
    house_mask_own = (
        df['rawhn'].isna() &
        df['street'].notna() & 
        (df['ownershp'] == 10) &
        (prev_page == df['pageno']) &
        (prev_street == df['street'])
    )
    print(prev_rawhn[house_mask_own])
    df.loc[house_mask_own, 'rawhn'] = (prev_rawhn[house_mask_own].values + 2)
    print('interpolated missing house numbers for owners')

    # additional adjustments to make matching easier
    df['street'] = (df['street']
        .str.lower()
        .str.replace(r'\bavenue\b', 'ave', regex=True)
        .str.replace(r'\bstreet\b', '', regex=True)
        .str.replace(r'\broad\b', ' rd', regex=True)
        .str.replace(r'\bdrive\b', 'dr', regex=True)
        .str.replace(r'\bplace\b', 'pl', regex=True)
        .str.replace(r'\bcourt\b', ' ct', regex=True)
        .str.replace(r'( \w )', '', regex=True)
    )

    # make sure nan is correctly coded as missing
    df['street'] = df['street'].replace("nan", np.nan)
    return df

### FUNCTION TO MATCH ADDRESSES TO KNOWN STREETS FROM STEVE MORSE IN 3 ROUNDS ###
def match_addresses(df, streets):
    known_streets = streets['street'].str.lower().unique()
    df['street_match'] = pd.Series(np.nan, index=df.index, dtype='object')
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
    print(df['street_match'].notna().sum(), 'streets matched in round 1')

    # round 2: fuzzy match street to previous value
    # rationale - streets adjacent in the census are likely to be adjacent geographically
    while True:
        prev_match = df['street_match'].shift(1)
        sim = df.apply(
            lambda row: distance.JaroWinkler.normalized_similarity(row['street'], row['prev_street'])
            if pd.notna(row['street']) and pd.notna(row['prev_street']) else 0, axis=1
        )
        similar_adjacent_mask = (
            df['street_match'].isna() &
            df['street'].notna() &
            (sim >=0.8)
            )
        if similar_adjacent_mask.any():
            df.loc[similar_adjacent_mask, 'street_match'] = prev_match[similar_adjacent_mask].values
        else:
            break
    print(df['street_match'].notna().sum(), 'streets matched in round 2')
    df.drop(columns=['prev_street'], inplace=True)

    # round 3: fuzzy match streets within the same page
    # rationale - same as above but expanding our pool slightly 
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    def round_3(row):
        candidates = df.loc[
            (df['pageno'] == row['pageno']) &
            (df['street'].notna()) &
            (df.index != row.name), 'street'
        ].unique()
        if len(candidates) == 0:
            return np.nan
        match = process.extractOne(
            row['street'], candidates,
            scorer = distance.JaroWinkler.normalized_similarity,
            score_cutoff=0.8
            )
        if match is not None:
            best_candidate = match[0]
            match_known = process.extractOne(
                best_candidate, known_streets,
                scorer = distance.JaroWinkler.normalized_similarity,
                score_cutoff=0.2
            )
            return match_known[0] if match_known is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_3, axis=1)
    print(df['street_match'].notna().sum(), 'streets matched in round 3')

    # round 4: fuzzy match any remaining observations
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    def round_4(row):
            match = process.extractOne(
                row['street'], known_streets,
                scorer = distance.JaroWinkler.normalized_similarity,
                score_cutoff=0.2
            )
            return match[0] if match is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_4, axis=1)
    print(df['street_match'].notna().sum(), 'streets matched in round 4')

    # fill in remaining unmatched streets with original value
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched, 'street']

    return df

####################################################################################################

if not os.path.exists('data/output/ga_streets.csv'):
    from scrape_streets import street_list
else:
    street_list = pd.read_csv('data/output/ga_streets.csv')

ga = pd.read_csv('data/output/census_ga_1940.csv')

# keep only columns and counties we need
cols = ['valueh', 'race', 'street', 'city', 'urban', 'countyicp', 'stateicp', 'rent', 
        'enumdist', 'respond', 'numperhh', 'numprec', 'serial', 'rawhn', 'ownershp', 'pageno', 'dwelling']
ga2 = ga.loc[ga['countyicp'].isin([1210, 890]), cols]
atl = ga2[ga2['city'] == 350].copy()

# recode valueh and rent missing values
atl['valueh'] = atl['valueh'].replace([9999998, 9999999], np.nan)
atl['rent'] = atl['rent'].replace([0, 9998, 9999], np.nan)

# keep one observation per household/serial 
atl = atl.drop_duplicates(subset = ['serial'], keep = 'first').reset_index()
print(f'number of records:{len(atl)}')

atl = clean_addresses(atl)
print('address cleaning done')

# print out how many valid streets we have before matching
print(f'rows with street info: {atl["street"].notna().sum()}')

atl = match_addresses(atl, street_list)
print('address matching done')
print(f'rows with street info: {atl["street_match"].notna().sum()}')

atl.to_csv('data/output/atl_cleaned.csv', index=False)
print('csv created')