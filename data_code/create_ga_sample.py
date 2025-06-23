import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.geometry

### FUNCTION TO CLASSIFY GRID BASED ON ZONING ###
def classify_grid(df):
    # reclassify grids by zoning type (similar approach Noelkel et al. (2020) HOLC classification)
    df['area'] = df['geometry'].area
    df['area_res'] = np.where(df['zoning'] == 'residential', df['area'], 0)
    df['area_ind'] = np.where(df['zoning'] == 'industrial', df['area'], 0)
    df = df.groupby('grid_id').agg({'area': 'sum', 'area_res': 'sum', 'area_ind': 'sum'})
    df['pct_res'] = df['area_res'] / df['area']
    df['pct_ind'] = df['area_ind'] / df['area']

    # classify based on relative percentages (may not be ideal)
    df['maj_zoning'] = np.where(df['pct_res'] > df['pct_ind'], 'residential', 'industrial')
    return df

### FUNCTION TO CREATE THE SAMPLE GRID ### 
def create_grid(df):
    # condense zoning to residential and industrial
    df['zoning'] = np.where(df['Zonetype'] == 'dwelling', 'residential', 
                              np.where(df['Zonetype'] == 'apartment', 'residential', 
                                       np.where(df['Zonetype'] == 'industrial', 'industrial', 
                                                np.where(df['Zonetype'] == 'business', 'industrial', 'nan'))))
    df = df[~df['zoning'].isin(['nan'])]
    
    # create grid of 0.5km x 0.5km squares overlaying zoning map
    a, b, c, d  = df.total_bounds
    step = 500

    grid = gpd.GeoDataFrame(geometry = [
    shapely.geometry.box(minx, miny, maxx, maxy)
    for minx, maxx in zip(np.arange(a, c, step), np.arange(a, c, step)[1:])
    for miny, maxy in zip(np.arange(b, d, step), np.arange(b, d, step)[1:])], crs = df.crs)

    # numeric id for each grid square to assist with aggregation
    grid['grid_id'] = range(1, len(grid) + 1)

    # overlay zoning map with grid squares and classify each square
    zoning = df.overlay(grid, how = 'identity')
    zoning = classify_grid(zoning)

    # give each grid square its majority zoning
    output = grid.merge(zoning[['maj_zoning']], left_on = 'grid_id', right_index = True)
    return output