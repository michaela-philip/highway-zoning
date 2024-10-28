import pandas as pd
import geopandas as gpd
import numpy as np

###1 - merge in enumeration district shapefiles to ipums data###
ed_gis = gpd.read_file('data/input/AtlantaGA40/AtlantaGA_ed40.shp')
ed_gis = ed_gis.to_crs('EPSG:3857')
ipums = pd.read_csv('data/output/ipums_ga.csv')

ipums['black'] = np.where(ipums['race'] == 2, 1, 0)
ipums['white'] = np.where(ipums['race'] == 1, 1, 0)

ipums['enumdist'] = ipums['enumdist'].astype(str).str[-3:]
ed_gis['enumdist'] = ed_gis['enumdist'].astype(str).str.zfill(3)

ipums_agg = ipums.groupby('enumdist')[['ownershp', 'rent', 'valueh', 'black', 'white', 'incwage']].mean().reset_index()
ed_gis.rename(columns={'ed': 'enumdist'}, inplace=True)

##now overlay digitized zoning map and highway maps whenever they exist 
zoning = gpd.read_file('data/input/1945zoning/1945zoning.shp')

def classify_ed(data, zoning):
    #polygons should be subsections of tracts that fall within one HOLC grade only
    polygons = data.overlay(zoning, how = 'identity', keep_geom_type = False)
    polygons['area'] = polygons['geometry'].area

    #assign each area measurement to its grade so that we can aggregate
    polygons['area_residential'] = np.where(polygons['Zonetype'] == 'Residential', polygons['area'], 0)
    polygons['area_sub_residential'] = np.where(polygons['Zonetype'] == 'Substandard Residential', polygons['area'], 0)
    polygons['area_industrial'] = np.where(polygons['Zonetype'] == 'Industrial', polygons['area'], 0)
    polygons['area_public'] = np.where(polygons['Zonetype'] == 'Public', polygons['area'], 0)
    polygons['area_schools'] = np.where(polygons['Zonetype'] == 'Schools/Parks', polygons['area'], 0)
    polygons['area_uncategorized'] = np.where(~polygons['Zonetype'].isin(['Residential', 'Substandard Residential', 'Industrial', 'Public', 'Schools/Parks']), polygons['area'], 0)  
    
    if not (polygons['area_residential'] + polygons['area_sub_residential'] + polygons['area_industrial'] + polygons['area_public'] + polygons['area_schools'] + polygons['area_uncategorized']).sum() == polygons['area'].sum():
        raise ValueError(f"Sum of areas for ed {polygons['enumdist']} are incorrect")    
    
    #aggregate up to the enumeration district level, summing up total area in each grade and overall 
    ed = polygons.dissolve(by = 'enumdist', aggfunc='sum')

    #calculate percentage of each ed that belongs to each grade
    ed['pct_residential'] = ed['area_residential'] / ed['area']
    ed['pct_sub_residential'] = ed['area_sub_residential'] / ed['area']
    ed['pct_industrial'] = ed['area_industrial'] / ed['area']
    ed['pct_public'] = ed['area_public'] / ed['area']
    ed['pct_schools'] = ed['area_schools'] / ed['area']
    ed['pct_uncategorized'] = ed['area_uncategorized'] / ed['area']

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
            print(f"ED {row['enumdist']} doesn't fit a category because Residential = {row['pct_residential']}, Sub Residential = {row['pct_sub_residential']}, Industrial = {row['pct_industrial']}, Public = {row['pct_public']},  Schools = {row['pct_school']}, Uncategorized = {row['pct_uncategorized']}")
            return None
        
    ed['zone'] = ed.apply(assign_grade_6, axis = 1)
    return ed

ed = classify_ed(ed_gis, zoning)

ipums_ed = ipums_agg.merge(ed, on = 'enumdist', how = 'right')
ipums_ed = gpd.GeoDataFrame(ipums_ed, geometry='geometry')

#calculate summary statistics
ipums_ed['bpct'] = ipums_ed['bpop'] / ipums_ed['totalpop']
ipums_ed['immpct'] = ipums_ed['immpop'] / ipums_ed['totalpop']

ipums_ed[['bpct', 'immpct', 'meansei', 'incwage', 'ownershp', 'valueh']].describe()


###merge in 1952 hwy shapefile###
