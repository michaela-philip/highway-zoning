import os
import requests
import pandas as pd
import rasterio
from rasterio.merge import merge
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# data come from the following url
# https://apps.nationalmap.gov/downloader/ 1/3 arc second elevation data for each city
# downloaded and links aggregated into 'input/{city}/geology/data.txt'

REQUEST_TIMEOUT = (10, 120)

# function to fetch and download each file from the list of urls in data.txt
def download_file(source_url, target_dir):
    destination_name = os.path.basename(source_url)
    destination_path = os.path.join(target_dir, destination_name)
    temp_path = destination_path + ".tmp"

    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(destination_path):
        return True, f"Already exists: {destination_path}", source_url

    try:
        with requests.get(source_url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            
            # Get expected file size from server if available
            content_length = response.headers.get("content-length")
            bytes_written = 0

            with open(temp_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output_file.write(chunk)
                        bytes_written += len(chunk)

            # Verify we got the expected amount
            if content_length:
                expected_size = int(content_length)
                if bytes_written != expected_size:
                    os.remove(temp_path)
                    raise RuntimeError(
                        f"Incomplete download: expected {expected_size} bytes, got {bytes_written}"
                    )

        # Atomic move: only rename if download is complete
        os.rename(temp_path, destination_path)
        return True, f"Downloaded: {destination_path}", source_url

    except Exception as exc:
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False, f"Failed to download {source_url}: {exc}", source_url
    
# perform download for each city in sample
def get_geology(sample):
    for city in sample['city'].unique():
        raster = f"data/input/{city}/geology/topology.tif"
        if os.path.exists(raster):
            print(f"Raster already exists for {city}, skipping download.")
            continue
        urls_file = f"data/input/{city}/geology/data.txt"
        with open(urls_file) as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            raise RuntimeError(f"No URLs found in {urls_file}")

        out_dir = f"data/input/{city}/geology/rasters"
        print(f"{city}: downloading {len(urls)} file(s)...")

        failures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_file, url, out_dir) for url in urls]
            for future in as_completed(futures):
                ok, msg, source_url = future.result()
                print(msg)
                if not ok:
                    failures.append(source_url)

        if failures:
            raise RuntimeError(f"{city}: {len(failures)} download(s) failed.")

        print(f"{city} geology downloaded")

# aggregate rasters into mosaic
def create_mosaic(sample):
    for city in sample['city'].unique():
        raster = f"data/input/{city}/geology/topology.tif"
        if os.path.exists(raster):
            print(f"Raster already exists for {city}, skipping mosaicking.")
            continue
        out_dir = f"data/input/{city}/geology/rasters"
        os.makedirs(out_dir, exist_ok=True)

        # aggregate into one mosaic tiff
        tiles = glob.glob(os.path.join(out_dir, '*.tif'))
        if not tiles:
            raise RuntimeError(f"No .tif files found for mosaicking in {city}.")
        src_files_to_mosaic = [rasterio.open(f) for f in tiles]
        mosaic, out_trans = merge(src_files_to_mosaic)

        # update metadata
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update({
            'height':mosaic.shape[1],
            'width':mosaic.shape[2],
            'transform': out_trans
        })
        for src in src_files_to_mosaic:
            src.close()
        
        out_path = os.path.join(f"data/input/{city}/geology", "topology.tif")
        with rasterio.open(out_path, 'w', **out_meta) as dest:
            dest.write(mosaic)

        for tif in tiles:
            try:
                os.remove(tif)
                print(f"Deleted {tif}")
            except Exception as e:
                print(f"Error deleting {tif}: {e}")

# ####################################################################################################
# ### SECTION TO BE EDITED UPON ADDITION OF NEW CITIES ###
values = [
    ('atlanta', 'AT', 'georgia', 'GA', 44, [1210, 890], 350),
    ('louisville', 'LO', 'kentucky', 'KY', 51, [1110], 3750),
    ('littlerock', 'LR', 'arkansas', 'AR', 5, [1190], 3650)]
keys=['city', 'cityabbr', 'state', 'stateabbr', 'stateicp', 'countyicp', 'cityicp']
rows = [dict(zip(keys, v)) for v in values]
sample = pd.DataFrame(rows)
# ####################################################################################################

# sample = pd.read_pickle('data/input/samplelist.pkl')
get_geology(sample)
create_mosaic(sample)