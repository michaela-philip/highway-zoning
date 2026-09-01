import os
import time
import pandas as pd
import numpy as np

from geocode_utils import as_xy, distance_meters, query_single_line_matches, pick_closest_candidate

## Addresses that geocode successfully in clean.py sometimes still fall outside every
## grid square once create_sample.py lays the study-area grid down. create_sample.py
## recovers a few of these by checking whether the two enumeration-adjacent neighbors
## land in the same grid cell, but anything it can't recover is silently dropped.
##
## The working theory (see clean.py's Part A flagging) is that a meaningful share of
## these are hidden single-line ties: the batch geocoder returned one candidate where
## the single-line geocoder would have returned two, and picked the wrong one -- often
## when the enumerator crosses from one street to the next. This script pulls the
## dropped serials back out, re-queries the single-line geocoder for each one, and
## picks whichever candidate lands closest to the record's enumeration neighbors.
##
## Designed to run against a potentially large, network-bound worklist without losing
## progress: results are checkpointed to disk after every chunk, so a crash or manual
## interruption only costs wall-clock time, and a repeat of the whole address book
## isn't needed to pick back up later.

GRIDSIZE_FOR_IDENTIFICATION = 150  # finest grid -- a point outside it is outside every coarser one too
CHECKPOINT_PATH = 'data/intermed/resolved_dropped_geocodes.pkl'
GEOCODED_DATA_PATH = 'data/intermed/geocoded_data.pkl'
CHUNK_SIZE = 200
CONSECUTIVE_FAILURE_LIMIT = 20  # abort the run rather than grind through a network outage row by row


### FUNCTION TO FIND CENSUS RECORDS THAT GEOCODED SUCCESSFULLY BUT NEVER GOT A GRID_ID ###
def find_unrecovered_serials(census, sample):
    from create_sample import build_grid, assign_grid_ids, get_primary_zoning

    unrecovered = []
    for city in sample['city'].unique():
        city_df = census[census['city'] == city].copy()
        city_zoning = get_primary_zoning(city)
        grid = build_grid(city_zoning, GRIDSIZE_FOR_IDENTIFICATION)
        with_grid, _, _ = assign_grid_ids(city_df, grid)
        missing = with_grid[with_grid['grid_id'].isna() & with_grid['coordinates'].notna()]
        unrecovered.append(missing)
    if not unrecovered:
        return pd.DataFrame(columns=census.columns)
    return pd.concat(unrecovered, ignore_index=True)


### FUNCTIONS FOR A RESUMABLE, CRASH-SAFE CHECKPOINT FILE ###
def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return pd.read_pickle(CHECKPOINT_PATH)
    return pd.DataFrame(columns=['serial', 'status', 'coordinates'])


def save_checkpoint(checkpoint_df):
    tmp_path = CHECKPOINT_PATH + '.tmp'
    checkpoint_df.to_pickle(tmp_path)
    os.replace(tmp_path, CHECKPOINT_PATH)  # atomic, so a crash mid-write can't corrupt the checkpoint


### FUNCTION TO RE-QUERY ONE DROPPED RECORD AND CLASSIFY THE OUTCOME ###
def resolve_row(row, prev_coordinate, next_coordinate):
    candidates = query_single_line_matches(row['address'], row['city'], row['state'], row['zipcode'])
    if candidates is None:
        return 'error', None
    if len(candidates) == 0:
        return 'no_match', None

    chosen = pick_closest_candidate(candidates, [prev_coordinate, next_coordinate])
    original = as_xy(row['coordinates'])
    same_as_original = (
        len(candidates) == 1 and original is not None and
        distance_meters(chosen, original) is not None and distance_meters(chosen, original) < 1
    )
    status = 'confirmed_outside_grid' if same_as_original else 'resolved'
    return status, chosen


### FUNCTION TO WORK THROUGH THE DROPPED-SERIAL WORKLIST, CHECKPOINTING AS IT GOES ###
def resolve_dropped_geocodes(census_full, unrecovered, chunk_size=CHUNK_SIZE):
    checkpoint = load_checkpoint()
    already_done = set(checkpoint['serial'])
    todo = unrecovered[~unrecovered['serial'].isin(already_done)].reset_index(drop=True)
    print(f'{len(already_done)} already resolved from a previous run; {len(todo)} remaining')
    if len(todo) == 0:
        return checkpoint

    # positional lookup so neighbor coordinates are O(1) per row rather than a linear scan
    census_full = census_full.reset_index(drop=True)
    pos_lookup = pd.Series(census_full.index.values, index=census_full['serial'].values)
    coords_array = census_full['coordinates'].apply(as_xy).to_numpy()

    consecutive_failures = 0
    new_results = []
    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        for _, row in chunk.iterrows():
            pos = pos_lookup.get(row['serial'])
            prev_coordinate = coords_array[pos - 1] if pos is not None and pos > 0 else None
            next_coordinate = coords_array[pos + 1] if pos is not None and pos < len(coords_array) - 1 else None

            status, coord = resolve_row(row, prev_coordinate, next_coordinate)
            new_results.append({'serial': row['serial'], 'status': status, 'coordinates': coord})

            consecutive_failures = consecutive_failures + 1 if status == 'error' else 0
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                print(f'{CONSECUTIVE_FAILURE_LIMIT} consecutive failures -- aborting run, progress saved')
                checkpoint = pd.concat([checkpoint, pd.DataFrame(new_results)], ignore_index=True)
                save_checkpoint(checkpoint)
                return checkpoint
            time.sleep(1)

        checkpoint = pd.concat([checkpoint, pd.DataFrame(new_results)], ignore_index=True)
        save_checkpoint(checkpoint)
        new_results = []
        print(f'checkpointed {len(checkpoint)} / {len(already_done) + len(todo)} total')
        time.sleep(2)

    return checkpoint


### FUNCTION TO PATCH RESOLVED COORDINATES BACK INTO geocoded_data.pkl ###
def apply_resolved_coordinates(geocoded_path=GEOCODED_DATA_PATH):
    census = pd.read_pickle(geocoded_path)
    checkpoint = load_checkpoint()
    resolved = checkpoint[checkpoint['status'] == 'resolved'].set_index('serial')['coordinates']

    census = census.set_index('serial')
    update_idx = census.index.intersection(resolved.index)
    census.loc[update_idx, 'coordinates'] = resolved.loc[update_idx].apply(list)
    census = census.reset_index()

    tmp_path = geocoded_path + '.tmp'
    census.to_pickle(tmp_path)
    os.replace(tmp_path, geocoded_path)
    print(f'{len(update_idx)} coordinates patched into {geocoded_path}')

    status_counts = checkpoint['status'].value_counts()
    print(f'checkpoint summary:\n{status_counts}')


if __name__ == '__main__':
    census = pd.read_pickle(GEOCODED_DATA_PATH)
    sample = pd.read_pickle('data/input/samplelist.pkl')

    unrecovered = find_unrecovered_serials(census, sample)
    print(f'{len(unrecovered)} geocoded records fall outside every grid cell')

    resolve_dropped_geocodes(census, unrecovered)
    apply_resolved_coordinates()
