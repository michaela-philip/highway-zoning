import pandas as pd
import numpy as np
from rapidfuzz import process, distance, fuzz
import os
import re
import censusbatchgeocoder
import requests
import geopy, geopy.distance

# this is a beast of a file - overview below
# section 1: functions to clean, standardize, and match addresses
# section 2: master function to call functions defined in section 1
# section 3: ONLY section to be edited upon addition of new cities - add to 'sample' dataframe and run

####################################################################################################
### FUNCTION TO CLEAN AND INTERPOLATE ADDRESSES ###
def clean_addresses(df):
    # extract any additional information in () from house number
    df['street'] = df['street'].astype(str)
    df[['hotelinfo', 'street', 'streetinfo']] = df['street'].str.extract(
        r'^(?:(?P<hotelinfo>[\w\s]+hotel)\s+)?(?P<street>[^\(]+?)(?:\s*\((?P<streetinfo>[^)]*)\))?\s*$',
        flags=re.IGNORECASE)
    
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
        prev_city = df['city'].shift(1)
        next_city = df['city'].shift(-1)
        
        street_interp_forward_mask = (
            df['street'].isna() &
            df['rawhn'].notna() &
            prev_rawhn.notna() &
            prev_street.notna() &
            (prev_city == df['city']) &
            (prev_page == df['pageno']) & 
            ((prev_rawhn - df['rawhn']).abs() <= 6) # norm from PVC/Logan and Zhang (2019)
            )
        street_interp_back_mask = (
            df['street'].isna() &
            df['rawhn'].notna() &
            next_rawhn.notna() &
            next_street.notna() &
            (next_city == df['city']) &
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

    # make sure nan is correctly coded as missing
    df['street'] = df['street'].replace("nan", np.nan)
    df['rawhn'] = df['rawhn'].replace("nan", np.nan)
    return df

### FUNCTION TO STANDARDIZE ADDRESSES ###
# two goals here - maximize matches to known streets from Steve Morse and matches to geocoding locations from US Census
def standardize_addresses(df):
    # abbreviate street types to match steve morse, remove directional notations
    df['street'] = (df['street']
        .str.lower()
        .str.replace(r'\bavenue\b', 'ave', regex=True)
        .str.replace(r'\bstreet\b', '', regex=True)
        .str.replace(r'\broad\b', ' rd', regex=True)
        .str.replace(r'\bdrive\b', 'dr', regex=True)
        .str.replace(r'\bplace\b', 'pl', regex=True)
        .str.replace(r'\bcourt\b', ' ct', regex=True)
        .str.replace(r'-', '', regex=True)
        .str.strip()
    )
    # save directions separately (steve morse doesn't include them) - can reintroduce for geocoding (census api does)
    df[['street', 'street_direction']] = df['street'].str.extract(
        r'^(?P<street>.*?)(?:\s+(?P<street_direction>nw|ne|sw|se|n|s|e|w))?$', flags=re.IGNORECASE)
    df['street_direction'] = df['street_direction'].str.strip().fillna('')
    # replace ordinal words with numbers
    ordinal_map = {
        'first': '1st',
        'second': '2nd',
        'third': '3rd',
        'fourth': '4th',
        'fifth': '5th',
        'sixth': '6th',
        'seventh': '7th',
        'eighth': '8th',
        'ninth': '9th',
        'tenth': '10th',
        'eleventh': '11th',
        'twelfth': '12th',
        'thirteenth': '13th',
        'fourteenth': '14th',
        'fifteenth': '15th',
        'sixteenth': '16th',
        'seventeenth': '17th',
        'eighteenth': '18th',
        'nineteenth': '19th',
        'twentieth': '20th'
    }
    # do a fuzzy replacement to take care of typos in ordinal words
    def fuzzy_replace_ordinals(street):
        if pd.isna(street):
            return street
        words = street.split()
        for i, word in enumerate(words):
            for key in ordinal_map:
                sim = distance.JaroWinkler.normalized_similarity(word, key)
                if sim > 0.8:
                    words[i] = ordinal_map[key]
                    break
        return ' '.join(words)

    df['street'] = df['street'].apply(fuzzy_replace_ordinals)
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
            scorer = fuzz.ratio, 
            score_cutoff= 100)
        return match[0] if match is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched, 'street'].apply(round_1)
    print(df['street_match'].notna().sum(), 'streets matched in round 1')

    # round 2: fuzzy match street to previous value
    # rationale - streets adjacent in the census are likely to be adjacent geographically
    iter_count = 0
    max_iter = 30
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
                print(iter_count)
                df.loc[similar_adjacent_mask, 'street_match'] = prev_match[similar_adjacent_mask].values
        else:
            break
        iter_count += 1
        print(df['street_match'].notna().sum(), iter_count)
        if iter_count > max_iter:
            print('max iterations reached')
            break
    print(df['street_match'].notna().sum(), 'streets matched in round 2')
    df.drop(columns=['prev_street'], inplace=True)

    # round 3: fuzzy match streets within the same page
    # rationale - same as above but expanding our pool slightly 
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    page_candidates = {
        pageno: group['street'].dropna().unique()
        for pageno, group in df.groupby('pageno')
        }

    def round_3(row):
        candidates = page_candidates.get(row['pageno'], [])
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
                scorer = distance.DamerauLevenshtein.normalized_similarity,
                score_cutoff=0.3
            )
            return match_known[0] if match_known is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_3, axis=1)
    print(df['street_match'].notna().sum(), 'streets matched in round 3')

    # round 4: fuzzy match any remaining observations
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    def round_4(row):
            match = process.extractOne(
                row['street'], known_streets,
                scorer = distance.DamerauLevenshtein.normalized_similarity,
                score_cutoff=0.3
            )
            return match[0] if match is not None else np.nan
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched].apply(round_4, axis=1)
    print(df['street_match'].notna().sum(), 'streets matched in round 4')

    # fill in remaining unmatched streets with original value
    mask_unmatched = df['street_match'].isna() & df['street'].notna()
    df.loc[mask_unmatched, 'street_match'] = df.loc[mask_unmatched, 'street']
    df.drop(columns = ['rawhninfo', 'hotelinfo', 'streetinfo'], inplace=True)

    # match new street names to street names that have been changed since 1940 to assist with geocoding
    street_changes = pd.read_csv('data/output/atl_street_changes.csv')
    street_changes['old_name'] = street_changes['old_name'].str.replace(r'\([^()]*\)', '', regex=True).str.strip()
    def new_names(street_match):
        match = process.extractOne(street_match, street_changes['old_name'],
            scorer = distance.DamerauLevenshtein.normalized_similarity,
            score_cutoff=0.7)
        return match[0] if match is not None else np.nan
    df['old_name'] = df['street_match'].apply(new_names)
    df = df.merge(street_changes, on = 'old_name', how = 'left')

    print(df['new_name'].notna().sum(), 'streets matched to street changes')
    print(len(df['serial'].unique())) 

    return df

### FUNCTION TO CALL MATCHING FUNCTION CITY BY CITY ###
def match_addresses_citywide(df, sample, city_street_lists):
    results = []
    for city in sample['city'].unique():
        city_df = df[df['city'].str.lower() == city].copy()
        streets = city_street_lists[city]
        matched = match_addresses(city_df, streets)
        results.append(matched) 
    # concat and return
    return pd.concat(results, ignore_index = True)

### FUNCTION TO INTERPOLATE HOUSE NUMBERS ### 
def interpolate_house_numbers(df):
    # interpolate missing house numbers for renters - use previous house number
    print(df['rawhn'].notna().sum())
    prev_page = df['pageno'].shift(1)
    prev_street = df['street_match'].shift(1)
    prev_rawhn = df['rawhn'].shift(1)

    house_mask_rent = (
        df['rawhn'].isna() &
        df['street_match'].notna() & 
        (df['ownershp'] != 10) &
        (prev_page == df['pageno']) &
        (prev_street == df['street_match'])
    )
    print(prev_rawhn[house_mask_rent])
    df.loc[house_mask_rent, 'rawhn'] = prev_rawhn[house_mask_rent].values
    print('interpolated missing house numbers for renters')
    
    # interpolate missing house numbers for owners - add 2 to previous house number
    prev_page = df['pageno'].shift(1)
    prev_street = df['street_match'].shift(1)
    prev_rawhn = df['rawhn'].shift(1)
    
    house_mask_own = (
        df['rawhn'].isna() &
        df['street_match'].notna() & 
        (df['ownershp'] == 10) &
        (prev_page == df['pageno']) &
        (prev_street == df['street_match'])
    )
    print(prev_rawhn[house_mask_own])
    df.loc[house_mask_own, 'rawhn'] = (prev_rawhn[house_mask_own].values + 2)

    # make sure nan is correctly coded as missing
    df['street'] = df['street'].replace("nan", np.nan)
    df['rawhn'] = df['rawhn'].replace("nan", np.nan)
    print('interpolated missing house numbers for owners', df['rawhn'].notna().sum())
    return df

### FUNCTION TO GEOCODE ADDRESSES ###
def geocode_addresses(df_orig, city_sample):
    # minor restructuring per geocoder requirements
    df = df_orig.copy()
    df = df.dropna(subset = ['rawhn', 'street_match'])
    df['new_name'] = df['new_name'].str.strip()
    df['rawhn'] = df['rawhn'].astype(str).str.replace('.0', '', regex=False).str.strip()
    df = df.dropna(subset = ['rawhn', 'street_match'])

    df['address'] = np.where(df['new_name'].notna(), 
                    df['rawhn'].astype(str).str.cat([df['new_name'].str.lower(), df['street_direction'].str.lower()], sep = ' ', na_rep = ''),
                    df['rawhn'].astype(str).str.cat([df['street_match'].str.lower(), df['street_direction'].str.lower()], sep = ' ', na_rep = ''))
    df['city'] = city_sample['city'] 
    df['state'] = city_sample['state'] 
    df['zipcode'] = ''
    df['id'] = df['serial'].astype(str)
    df = df[['id', 'address', 'city', 'state', 'zipcode']]

    # geocode using censusbatchgeocoder
    print('starting geocoding')
    third1, third2= int(len(df)/3), (2 * int(len(df)/3))
    d1 = df.iloc[:third1]
    d2 = df.iloc[(third1+1):(third2)]
    d3 = df.iloc[(third2 + 1):]
    result1 = pd.DataFrame(censusbatchgeocoder.geocode(d1.to_dict(orient = 'records'), zipcode = None))
    print('first done')
    result2 = pd.DataFrame(censusbatchgeocoder.geocode(d2.to_dict(orient = 'records'), zipcode = None))
    print('second done')
    result3 = pd.DataFrame(censusbatchgeocoder.geocode(d3.to_dict(orient = 'records'), zipcode = None))
    print('third done')
    geocoded_df = pd.concat([result1, result2, result3])
    geocoded_df.to_pickle('data/input/atl_geocoded.pkl')
    print(f"{geocoded_df['is_exact'].notna().sum()} records geocoded")
    print(f"{(geocoded_df['is_match'] == 'Tie').sum()} ties")
    print(f"{(geocoded_df['is_match'] == 'No_Match').sum()} unmatched")

    # want to deal with ties by choosing the coordinate closest to the adjacent entry 
    def resolve_ties(row):
        url = 'https://geocoding.geo.census.gov/geocoder/locations/address'
        params = {
            'street':row['address'],
            'city':row['city'],
            'state':row['state'],
            'zip':row['zipcode'],
            'benchmark':'4',
            'format':'json'
        }
        response = requests.get(url, params=params)
        prev_coordinate = row['prev_coordinate']
        if response.status_code == 200:
            matches = response.json().get('result', {}).get('addressMatches', [])
            if len(matches) > 0:
                c1 = (matches[0]['coordinates']['x'], matches[0]['coordinates']['y'])
                c2 = (matches[1]['coordinates']['x'], matches[1]['coordinates']['y'])
                # calculate distances between each set of points
                d1 = geopy.distance.distance((c1[1], c1[0]), (prev_coordinate[1], prev_coordinate[0])).meters
                d2 = geopy.distance.distance((c2[1], c2[0]), (prev_coordinate[1], prev_coordinate[0])).meters
                return c1 if d1 < d2 else c2    
        else:
            return None

    iter_count = 0
    max_iter = 10
    geocoded_df['coordinates'] = geocoded_df['coordinates'].str.split(',')
    while True:
        geocoded_df['prev_coordinate'] = geocoded_df['coordinates'].shift(1)
        candidates = ((geocoded_df['is_match'] == 'Tie') & geocoded_df['prev_coordinate'].notna() & geocoded_df['coordinates'].isna())
        # since this is based on the previous coordinate, I want this to iterate as long as possible
        if candidates.any():
            geocoded_df.loc[candidates, 'coordinates'] = geocoded_df.loc[candidates].apply(resolve_ties, axis=1)
            iter_count += 1
            print(f'iter {iter_count}: {geocoded_df['coordinates'].notna().sum()} coordinates')
        else:
            break
        if iter_count > max_iter:
            print('max iterations reached')
            break
    
    merged = df_orig.copy()
    merged['serial'] = merged['serial'].astype(str)
    merged = pd.merge(merged, geocoded_df, left_on='serial', right_on = 'id', how = 'left')
    return merged

### FUNCTION TO GEOCODE ADDRESSES CITY BY CITY ###
def geocode_addresses_citywide(df, sample):
    results = []
    for city in sample['city'].unique():
        city_df = df[df['city'].str.lower() == city].copy()
        city_sample = sample[sample['city'] == city].iloc[0]
        geocoded = geocode_addresses(city_df, city_sample)
        results.append(geocoded) 
    # concat and return
    return pd.concat(results, ignore_index = True)
####################################################################################################

####################################################################################################
### MASTER FUNCTION ###
def clean_data(census, sample, city_street_lists):
    cols = ['valueh', 'race', 'street', 'city', 'urban', 'countyicp', 'stateicp', 'rent', 
        'enumdist', 'respond', 'numperhh', 'numprec', 'serial', 'rawhn', 'ownershp', 'pageno', 'dwelling']

    mask = census['countyicp'].isin(sample['countyicp']) | census['cityicp'].isin(sample['cityicp'])
    df = census.loc[mask, cols]

    # recode valueh and rent missing values
    df['valueh'] = df['valueh'].replace([9999998, 9999999], np.nan)
    df['rent'] = df['rent'].replace([0, 9998, 9999], np.nan)
    df['black'] = np.where(df['race'] == 200, 1, 0)

    # keep one observation per household/serial 
    df = df.drop_duplicates(subset = ['serial'], keep = 'first').reset_index()
    print(f'number of records:{len(df)}')

    df = clean_addresses(df)
    print('address cleaning done')

    df = standardize_addresses(df)
    print('addresses standarized')

    df = match_addresses_citywide(df, sample, city_street_lists)
    print('address matching done')

    df = interpolate_house_numbers(df)
    print('house numbers interpolated')

    # pickle incase geocoding fails - no need to repeat entire process
    df.to_pickle('data/input/cleaned_data.pkl')    
    print('pickle created')

    df = geocode_addresses_citywide(df, sample)
    return df
####################################################################################################

####################################################################################################
### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
# list cities in sample
rows = [
    ('atlanta', 'georgia', 44, 1210, 350),
    ('atlanta', 'georgia', 44,  890, 350),
    ('louisville', 'kentucky', 51, 1110, 3750)]

sample = pd.DataFrame(rows, columns=['city', 'state', 'stateicp', 'countyicp', 'cityicp'])

# pull in street lists for each city
city_street_lists = {}
for city in sample['city'].unique():
    csv_path = f'data/input/{city}_streets.csv'
    if not os.path.exists(csv_path):
        from scrape_streets import atlanta_streets, louisville_streets
        if city == 'atlanta':
            city_street_lists[city] = atlanta_streets
        elif city == 'louisville':
            city_street_lists[city] = louisville_streets
        else:
            raise ValueError(f'Street list for {city} not found and no scraping function available.')
    else:
        city_street_lists[city] = pd.read_csv(csv_path)
####################################################################################################

census = pd.read_csv('data/input/census_1940.pkl')
df = clean_data(census, sample, city_street_lists)
df.to_pickle('data/input/geocoded_data.pkl')
print('geocoded data pickled')