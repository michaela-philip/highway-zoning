import os
import requests
import pandas as pd
import rasterio
from rasterio.merge import merge
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# function to fetch and download each file from the list of urls in data.txt
def download_file(url, out_dir):
    filename = os.path.join(out_dir, os.path.basename(url))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        return f"Already exists: {filename}"
    try:
        r = requests.get(url, stream = True)
        r.raise_for_status()
        with open(filename, 'wb') as f_out:
            for chunk in r.iter_content(chunk_size=8192):
                f_out.write(chunk)
        return f"Downloaded: {filename}"
    except Exception as e:
        return f"Failed to download {url}: {e}"

# perform download for each city in sample
def get_geology(sample):
    for city in sample['city'].unique():
        urls_file = f"data/input/{city}/geology/data.txt"
        with open(urls_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        out_dir = f"data/input/{city}/geology/rasters"
        with ThreadPoolExecutor(max_workers = 4) as executor:
            futures = [executor.submit(download_file, url, out_dir) for url in urls]
            for future in as_completed(futures):
                print(future.result())
        print(f"{city} geology downloaded")

# aggregate rasters into mosaic
def create_mosaic(sample):
    for city in sample['city'].unique():
        out_dir = f"data/input/{city}/geology/rasters"
        os.makedirs(out_dir, exist_ok=True)

        # aggregate into one mosaic tiff
        tiles = glob.glob(os.path.join(out_dir, '*.tif'))
        if not tiles:
            raise RuntimeError(f"No .tif files found for mosaicking in {city}.")
        src_files_to_mosaic = [rasterio.open(f) for f in tiles]
        mosaic, out_trans = merge(src_files_to_mosaic)
        for src in src_files_to_mosaic:
            src.close()
        
        # update metadata
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            'height':mosaic.shape[1],
            'width':mosaic.shape[2],
            'transform': out_trans
        })
        out_path = os.path.join(f"data/input/{city}/geology", "topology.tif")
        with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(mosaic)

        for tif in tiles:
            try:
                os.remove(tif)
                print(f"Deleted {tif}")
            except Exception as e:
                print(f"Error deleting {tif}: {e}")

####################################################################################################
### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
values = [
    ('atlanta', 'AT', 'georgia', 'GA', 44, [1210, 890], 350),
    ('louisville', 'LO', 'kentucky', 'KY', 51, [1110], 3750)]
keys=['city', 'cityabbr', 'state', 'stateabbr', 'stateicp', 'countyicp', 'cityicp']
rows = [dict(zip(keys, v)) for v in values]
sample = pd.DataFrame(rows)
####################################################################################################

get_geology(sample)
create_mosaic(sample)