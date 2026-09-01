import time
import requests
import geopy.distance

SINGLE_LINE_URL = 'https://geocoding.geo.census.gov/geocoder/locations/address'
BENCHMARK = '4'  # Public_AR_Current -- matches censusbatchgeocoder's default, so this isn't a vintage mismatch


def as_xy(coordinates):
    """Parse a coordinates value (a 2-item list/tuple of numbers or numeric strings)
    into a (float, float) tuple, or None if it isn't a valid pair."""
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        try:
            return (float(coordinates[0]), float(coordinates[1]))
        except (TypeError, ValueError):
            return None
    return None


def distance_meters(c1, c2):
    """Great-circle distance in meters between two (x, y) = (lon, lat) tuples."""
    if c1 is None or c2 is None:
        return None
    return geopy.distance.distance((c1[1], c1[0]), (c2[1], c2[0])).meters


def query_single_line_matches(address, city, state, zipcode, max_retries=3, delay=5):
    """Query the Census single-line geocoder for every candidate address match.

    Returns a list of (x, y) coordinate tuples -- possibly empty, meaning a confirmed
    no-match -- or None if the request failed after retries (an unresolved outcome,
    distinct from a confirmed no-match).
    """
    params = {
        'street': address, 'city': city, 'state': state, 'zip': zipcode,
        'benchmark': BENCHMARK, 'format': 'json'
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(SINGLE_LINE_URL, params=params, timeout=10)
            if response.status_code == 200 and response.text.strip():
                try:
                    matches = response.json().get('result', {}).get('addressMatches', [])
                except Exception as e:
                    print(f"JSON decode error: {e}")
                    return None
                return [(m['coordinates']['x'], m['coordinates']['y']) for m in matches]
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this row.")
                return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    return None


def pick_closest_candidate(candidates, references):
    """Given candidate (x, y) coordinates and one or more reference coordinates,
    return the candidate closest to its nearest reference. Falls back to the sole
    candidate when there's only one, or when no reference is usable."""
    if not candidates:
        return None
    refs = [r for r in references if r is not None]
    if len(candidates) == 1 or not refs:
        return candidates[0]

    def score(c):
        return min(distance_meters(c, r) for r in refs)

    return min(candidates, key=score)
