import pandas as pd
import geopandas as gpd
import numpy as np

ed_gis = gpd.read_file('data/input/AtlantaGA40/AtlantaGA_ed40.shp')
ed_gis = ed_gis.to_crs('EPSG:3857')
ipums = pd.read_csv('data/output/ipums_ga.csv')
zoning = gpd.read_file('data/input/1945zoning/1945zoning.shp')

#clean up ipums data
ipums['black'] = np.where(ipums['race'] == 2, 1, 0)
ipums['white'] = np.where(ipums['race'] == 1, 1, 0)
ipums['valueh'] = np.where(ipums['valueh'] == 9999999, np.nan, ipums['valueh']).astype(float)
ipums['valueh'] = np.where(ipums['valueh'] == 9999998, np.nan, ipums['valueh']).astype(float)

# aggregate ipums data and get population counts
ipums_agg = ipums.groupby('enumdist')[['ownershp', 'rent', 'valueh', 'black', 'white', 'incwage']].mean().reset_index()
ipums_pop = ipums.groupby('enumdist').size().reset_index(name='population')
# NOTE these numbers don't look good and I don't know why -> not using ipums data for right now


# function to classify enumeration districts by zoning area
def classify_ed(data, zoning):
    # polygons should be subsections of tracts that fall within one zone grade only
    polygons = data.overlay(zoning, how = 'identity', keep_geom_type = False)
    polygons['area'] = polygons['geometry'].area

    # assign each area measurement to its grade so that we can aggregate
    polygons['area_residential'] = np.where(polygons['Zonetype'] == 'Residential', polygons['area'], 0)
    polygons['area_sub_residential'] = np.where(polygons['Zonetype'] == 'Substandard Residential', polygons['area'], 0)
    polygons['area_industrial'] = np.where(polygons['Zonetype'] == 'Industrial', polygons['area'], 0)
    polygons['area_public'] = np.where(polygons['Zonetype'] == 'Public', polygons['area'], 0)
    polygons['area_schools'] = np.where(polygons['Zonetype'] == 'Schools/Parks', polygons['area'], 0)
    polygons['area_uncategorized'] = np.where(~polygons['Zonetype'].isin(['Residential', 'Substandard Residential', 'Industrial', 'Public', 'Schools/Parks']), polygons['area'], 0)  
    
    if not (polygons['area_residential'] + polygons['area_sub_residential'] + polygons['area_industrial'] + polygons['area_public'] + polygons['area_schools'] + polygons['area_uncategorized']).sum() == polygons['area'].sum():
        raise ValueError(f"Sum of areas for ed {polygons['enumdist']} are incorrect")    
    
    # ed statistics are at enumeration district level (even though polygons were smaller) so we don't want to double count
    agg_funcs = ({
        'totalpop' : 'first',
        'bpop' : 'first', 
        'wpop' : 'first',
        'bpct' : 'first',
        'meansei' : 'first',
        'medsei' : 'first',
        'immpop' : 'first',
        'immpct' : 'first',
        'area': 'sum',
        'area_residential': 'sum',
        'area_sub_residential': 'sum',
        'area_industrial': 'sum',
        'area_public': 'sum',
        'area_schools': 'sum',
        'area_uncategorized': 'sum'
        })

    # Aggregate using the defined functions
    ed = polygons.dissolve(by='ed', aggfunc=agg_funcs)

    # calculate percentage of each ed that belongs to each grade
    ed['pct_residential'] = ed['area_residential'] / ed['area']
    ed['pct_sub_residential'] = ed['area_sub_residential'] / ed['area']
    ed['pct_industrial'] = ed['area_industrial'] / ed['area']
    ed['pct_public'] = ed['area_public'] / ed['area']
    ed['pct_schools'] = ed['area_schools'] / ed['area']
    ed['pct_uncategorized'] = ed['area_uncategorized'] / ed['area']

    # drop all ed that are entirely uncategorized
    ed = ed[ed['pct_uncategorized'] != 1.00]

    # simple categorization - may be worth also having a more complex one
    # categorizing by highest percentage that is NOT uncategorized
    def assign_grade_5(row):
        if row['pct_residential'] > max(row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'residential'
        if row['pct_sub_residential'] > max(row['pct_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'sub_residential'
        if row['pct_industrial'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_public'], row['pct_schools']):
            return 'industrial'
        if row['pct_public'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_schools']):
            return 'public'
        if row['pct_schools'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_public']):
            return 'school'
        else:
            print(f"ED {row['ed']} doesn't fit a category because Residential = {row['pct_residential']}, Sub Residential = {row['pct_sub_residential']}, Industrial = {row['pct_industrial']}, Public = {row['pct_public']},  Schools = {row['pct_school']}, Uncategorized = {row['pct_uncategorized']}")
            return None
        
    # same but leaving in an uncategorized option
    def assign_grade_6(row):
        if row['pct_residential'] > max(row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'residential'
        if row['pct_sub_residential'] > max(row['pct_residential'], row['pct_industrial'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'sub_residential'
        if row['pct_industrial'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'industrial'
        if row['pct_public'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_uncategorized'], row['pct_schools']):
            return 'public'
        if row['pct_schools'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_uncategorized'], row['pct_public']):
            return 'school'
        if row['pct_uncategorized'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'uncategorized'
        else:
            print(f"ED {row['ed']} doesn't fit a category because Residential = {row['pct_residential']}, Sub Residential = {row['pct_sub_residential']}, Industrial = {row['pct_industrial']}, Public = {row['pct_public']},  Schools = {row['pct_school']}, Uncategorized = {row['pct_uncategorized']}")
            return None
        
    ed['zone'] = ed.apply(assign_grade_5, axis = 1)
    return ed

# rewrite this to work with updated zoning data
def classify_ed_update(data, zoning):
    # polygons should be subsections of tracts that fall within one zone grade only
    polygons = data.overlay(zoning, how = 'identity', keep_geom_type = False)
    polygons['area'] = polygons['geometry'].area

    # assign each area measurement to its grade so that we can aggregate
    polygons['area_dwelling'] = np.where(polygons['Zonetype'] == 'dwelling', polygons['area'], 0)
    polygons['area_apartment'] = np.where(polygons['Zonetype'] == 'apartment', polygons['area'], 0)
    polygons['area_industrial'] = np.where(polygons['Zonetype'] == 'industrial', polygons['area'], 0)
    polygons['area_business'] = np.where(polygons['Zonetype'] == 'business', polygons['area'], 0)
    polygons['area_uncategorized'] = np.where(~polygons['Zonetype'].isin(['dwelling', 'apartment', 'industrial', 'business']), polygons['area'], 0)  
    
    if not (polygons['area_dwelling'] + polygons['area_apartment'] + polygons['area_industrial'] + polygons['area_business'] + polygons['area_uncategorized']).sum() == polygons['area'].sum():
        raise ValueError(f"Sum of areas for ed {polygons['enumdist']} are incorrect")    
    
    # ed statistics are at enumeration district level (even though polygons were smaller) so we don't want to double count
    agg_funcs = ({
        'totalpop' : 'first',
        'bpop' : 'first', 
        'wpop' : 'first',
        'bpct' : 'first',
        'meansei' : 'first',
        'medsei' : 'first',
        'immpop' : 'first',
        'immpct' : 'first',
        'area': 'sum',
        'area_dwelling': 'sum',
        'area_apartment': 'sum',
        'area_industrial': 'sum',
        'area_business': 'sum',
        'area_uncategorized': 'sum'
        })

    # Aggregate using the defined functions
    ed = polygons.dissolve(by='ed', aggfunc=agg_funcs)

    # consolidate the categories
    ed['area_residential'] = ed['area_dwelling'] + ed['area_apartment']
    ed['area_industry'] = ed['area_industrial'] + ed['area_business']

    # calculate percentages
    ed['pct_residential'] = ed['area_residential'] / ed['area']
    ed['pct_industrial'] = ed['area_industry'] / ed['area']
    ed['pct_uncategorized'] = ed['area_uncategorized'] / ed['area']

    # # calculate percentage of each ed that belongs to each grade
    # ed['pct_dwelling'] = ed['area_dwelling'] / ed['area']
    # ed['pct_apartment'] = ed['area_apartment'] / ed['area']
    # ed['pct_industrial'] = ed['area_industrial'] / ed['area']
    # ed['pct_business'] = ed['area_business'] / ed['area']
    # ed['pct_uncategorized'] = ed['area_uncategorized'] / ed['area']

    # drop all ed that are entirely uncategorized
    ed = ed[ed['pct_uncategorized'] != 1.00]

    def assign_grade_2(row):
        if row['pct_residential'] > row['pct_industrial']:
            return 'residential'
        if row['pct_industrial'] > row['pct_residential']:
            return 'industrial'
        else:
            print(f"ED {row['ed']} doesn't fit a category because Residential = {row['pct_residential']}, Industrial = {row['pct_industrial']}")
            return None

    # simple categorization - may be worth also having a more complex one
    # categorizing by highest percentage that is NOT uncategorized
    def assign_grade_5(row):
        if row['pct_residential'] > max(row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'residential'
        if row['pct_sub_residential'] > max(row['pct_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'sub_residential'
        if row['pct_industrial'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_public'], row['pct_schools']):
            return 'industrial'
        if row['pct_public'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_schools']):
            return 'public'
        if row['pct_schools'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_public']):
            return 'school'
        else:
            print(f"ED {row['ed']} doesn't fit a category because Residential = {row['pct_residential']}, Sub Residential = {row['pct_sub_residential']}, Industrial = {row['pct_industrial']}, Public = {row['pct_public']},  Schools = {row['pct_school']}, Uncategorized = {row['pct_uncategorized']}")
            return None
        
    # same but leaving in an uncategorized option
    def assign_grade_6(row):
        if row['pct_residential'] > max(row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'residential'
        if row['pct_sub_residential'] > max(row['pct_residential'], row['pct_industrial'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'sub_residential'
        if row['pct_industrial'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_public'], row['pct_uncategorized'], row['pct_schools']):
            return 'industrial'
        if row['pct_public'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_uncategorized'], row['pct_schools']):
            return 'public'
        if row['pct_schools'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_uncategorized'], row['pct_public']):
            return 'school'
        if row['pct_uncategorized'] > max(row['pct_residential'], row['pct_sub_residential'], row['pct_industrial'], row['pct_public'], row['pct_schools']):
            return 'uncategorized'
        else:
            print(f"ED {row['ed']} doesn't fit a category because Residential = {row['pct_residential']}, Sub Residential = {row['pct_sub_residential']}, Industrial = {row['pct_industrial']}, Public = {row['pct_public']},  Schools = {row['pct_school']}, Uncategorized = {row['pct_uncategorized']}")
            return None
        
    ed['zone'] = ed.apply(assign_grade_2, axis = 1)
    return ed

ed = classify_ed(ed_gis, zoning)

# add more relevant variables
ed['poppct'] = (ed['totalpop'] / (ed['totalpop'].sum())) * 100
ed['maj_black'] = np.where(ed['bpct'] > 50, 1, 0) # majority black population
ed['med_black'] = np.where(ed['bpct'] > (ed['bpct'].median()), 1, 0) # black population above median 
ed['zone_min'] = np.where((ed['zone'] == 'residential') | (ed['zone'] == 'sub_residential') | (ed['zone'] == 'public'), 'Any Residential', 
                         np.where((ed['zone'] == 'school') | (ed['zone'] == 'uncategorized'), 'Other', ed['zone'])) # condensed zoning variable

# transform for consistency in tables
ed['pct_residential'] = 100 * ed['pct_residential']
ed['pct_industrial'] = 100 * ed['pct_industrial']
ed['pct_sub_residential'] = 100 * ed['pct_sub_residential']
ed['pct_public'] = 100 * ed['pct_public']
ed['pct_schools'] = 100 * ed['pct_schools']
ed['pct_uncategorized'] = 100 * ed['pct_uncategorized']

# merge in highway data as of 1952
# should eventually rewrite as function so that I can add in all hwy shapefiles
hwy52 = gpd.read_file('data/input/1952hwy/1952hwy.shp')
hwy57 = gpd.read_file('data/input/1957hwy/1957hwy.shp')

ed_hwy57 = gpd.sjoin(ed, hwy57, how = 'left', predicate = 'intersects')
ed_hwy57['Status'] = ed_hwy57['Status'].fillna('No Highway')
ed_hwy57['Projected'] = (np.where(ed_hwy57['Status'] == 'Projected', 1, 0)) * 100
ed_hwy57['Constructed'] = (np.where(ed_hwy57['Status'] == 'Constructed', 1, 0)) * 100

agg_funcs = {col: 'first' for col in ed_hwy57.columns if col not in ['Projected', 'Constructed']}
agg_funcs['Projected'] = 'max'
agg_funcs['Constructed'] = 'max'
ed_hwy57 = ed_hwy57.groupby(['ed']).agg(agg_funcs).reset_index()

ed_hwy57 = ed_hwy57.set_geometry('geometry')