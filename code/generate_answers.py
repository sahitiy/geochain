import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

import pandas as pd
import pycountry
import pycountry_convert as pc
import reverse_geocoder as rg
from haversine import haversine # Used for distance calculation

# --- Constants for Location and Driving ---
COUNTRY_CODE_TO_CITY_MAPILLIARY: Dict[str, Any] = {
    'JO': 'Amman', 'NL': 'Amsterdam', 'TH': 'Bangkok', 'DE': 'Berlin', 'HU': 'Budapest',
    'DK': 'Copenhagen', 'IN': 'Goa', 'FI': 'Helsinki', 'GB': 'London', 'PH': 'Manila',
    'AU': 'Melbourne', 'RU': 'Moscow', 'KE': 'Nairobi', 'CA': ['Ottawa', 'Toronto'],
    'FR': 'Paris', 'BR': 'Sao Paulo', 'JP': 'Tokyo', 'NO': 'Trondheim', 'CH': 'Zurich',
    'US': ['Austin', 'Boston', 'Phoenix', 'San Francisco'],
}

CITY_TO_LAT_LON: Dict[str, Tuple[float, float]] = {
    'Ottawa': (45.4215, -75.6972), 'Toronto': (43.65107, -79.347015),
    'Austin': (30.2672, -97.7431), 'Boston': (42.3601, -71.0589),
    'Phoenix': (33.4484, -112.0740), 'San Francisco': (37.7749, -122.4194),
    # Add other cities from COUNTRY_CODE_TO_CITY_MAPILLIARY if they are single entries
    'Amman': (31.9454, 35.9284), 'Amsterdam': (52.3676, 4.9041),
    'Bangkok': (13.7563, 100.5018), 'Berlin': (52.5200, 13.4050),
    'Budapest': (47.4979, 19.0402), 'Copenhagen': (55.6761, 12.5683),
    'Goa': (15.2993, 74.1240), # Example for Goa, specific city might vary
    'Helsinki': (60.1699, 24.9384), 'London': (51.5072, -0.1276),
    'Manila': (14.5995, 120.9842), 'Melbourne': (-37.8136, 144.9631),
    'Moscow': (55.7558, 37.6173), 'Nairobi': (-1.2921, 36.8219),
    'Paris': (48.8566, 2.3522), 'Sao Paulo': (-23.5505, -46.6333),
    'Tokyo': (35.6762, 139.6503), 'Trondheim': (63.4305, 10.3951),
    'Zurich': (47.3769, 8.5417),
}


CONTINENT_CODE_TO_NAME_MAP: Dict[str, str] = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
    "OC": "Oceania", "SA": "South America", "AN": "Antarctica"
}

LEFT_DRIVING_COUNTRY_CODES: Set[str] = {"GB", "IN", "AU", "JP", "TH", "ID", "MT", "CY", "IE", "KE", "SG", "NZ", "PK", "ZA"} # Expanded a bit

# --- Constants for Question IDs (used as keys in the output dict and for column names) ---
QUESTION_IDS: Dict[str, str] = {
    "HAS_WATER_TRANSPORT": "0", "HAS_LAND_TRANSPORT": "1", "HAS_TRAFFIC_LIGHT": "2", "HAS_FLAG": "3",
    "NEAR_EQUATOR": "4", "NEAR_POLE": "5", "IN_NORTHERN_HEMISPHERE": "6", "CONTINENT": "7",
    "DRIVING_SIDE": "8", "COUNTRY": "9", "NEAR_COAST": "10", "IS_ISLAND": "11",
    "IS_DESERT_REGION": "12", "IS_MOUNTAINOUS": "13", "CLIMATE_TYPE": "14", "IS_BIG_CITY": "15",
    "IS_SMALL_TOWN": "16", "LANGUAGES": "17", "STATE_PROVINCE": "18", "CITY": "19", "COORDINATES": "20"
}

# --- Helper Functions ---

def country_code_to_continent(country_code: str) -> str:
    """Converts a 2-letter country code to its continent name."""
    try:
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        return CONTINENT_CODE_TO_NAME_MAP.get(continent_code, "Unknown")
    except (KeyError, TypeError): # pycountry_convert can raise KeyError or TypeError for invalid codes
        return "Unknown"

def country_code_to_country_name(country_code: str) -> str:
    """Converts a 2-letter country code to its full country name."""
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        return country.name if country else "Unknown"
    except LookupError:
        return "Unknown"

def get_mapillary_city_for_coords(country_code: str, lat: float, lon: float) -> str:
    """Determines the Mapillary city name based on country code and coordinates."""
    if country_code not in COUNTRY_CODE_TO_CITY_MAPILLIARY:
        return "Unknown"

    city_or_cities = COUNTRY_CODE_TO_CITY_MAPILLIARY[country_code]
    if isinstance(city_or_cities, list):
        # Ensure all cities in the list are in CITY_TO_LAT_LON
        valid_cities_in_list = [c for c in city_or_cities if c in CITY_TO_LAT_LON]
        if not valid_cities_in_list:
            return "Unknown" # Or a default city from the list if appropriate

        distances = {
            city_name: haversine((lat, lon), CITY_TO_LAT_LON[city_name])
            for city_name in valid_cities_in_list
        }
        return min(distances, key=distances.get) if distances else "Unknown"
    else: # It's a single city name string
        return city_or_cities

def get_location_metadata(lat: float, lon: float) -> Dict[str, str]:
    """Retrieves location metadata (country code, city, state) using reverse geocoding."""
    try:
        geo_info = rg.search((lat, lon), mode=1)
        if not geo_info:
            return {"country_code": "Unknown", "city": "Unknown", "state": "Unknown"}
        
        result = geo_info[0]
        country_code = result.get("cc", "Unknown")
        # Determine city using our custom logic for Mapillary dataset
        city = get_mapillary_city_for_coords(country_code, lat, lon)
        state = result.get("admin1", "Unknown")
        
        return {"country_code": country_code, "city": city, "state": state}
    except Exception as e:
        print(f"Error in reverse geocoding for ({lat}, {lon}): {e}") # Consider using logging
        return {"country_code": "Unknown", "city": "Unknown", "state": "Unknown"}


def auto_answer_questions(
    lat: float,
    lon: float,
    per_city_answers_df: pd.DataFrame,
    image_classes: dict
) -> Dict[str, str]:
    """
    Generates answers to a predefined set of questions based on location and detected image classes.
    """
    meta = get_location_metadata(lat, lon)
    country = country_code_to_country_name(meta["country_code"])
    continent = country_code_to_continent(meta["country_code"])
    city = meta["city"]

    city_specific_answers_row: pd.Series
    if city != "Unknown" and not per_city_answers_df.empty:
        city_rows = per_city_answers_df[per_city_answers_df['City'].str.lower() == city.lower()]
        if not city_rows.empty:
            city_specific_answers_row = city_rows.iloc[0]
        else:
            city_specific_answers_row = pd.Series(dtype='object') # Empty series
    else:
        city_specific_answers_row = pd.Series(dtype='object')

    # Pre-calculate presence of key classes for efficiency
    has_boat_or_ship = ('boat' in image_classes or
                        'ship' in image_classes)
    has_land_vehicle = (
        'bus' in image_classes or
        'truck' in image_classes or
        'car' in image_classes or
        'van' in image_classes or
        'minibike, motorbike' in image_classes or
        'bicycle' in image_classes
    )
    has_traffic_light = 'traffic light' in image_classes
    has_flag = 'flag' in image_classes

    answers: Dict[str, str] = {
        QUESTION_IDS["HAS_WATER_TRANSPORT"]: "Yes" if has_boat_or_ship else "No",
        QUESTION_IDS["HAS_LAND_TRANSPORT"]: "Yes" if has_land_vehicle else "No",
        QUESTION_IDS["HAS_TRAFFIC_LIGHT"]: "Yes" if has_traffic_light else "No",
        QUESTION_IDS["HAS_FLAG"]: "Yes" if has_flag else "No",
        QUESTION_IDS["NEAR_EQUATOR"]: "Yes" if abs(lat) < 10 else "No",
        QUESTION_IDS["NEAR_POLE"]: "Yes" if abs(lat) > 66.5 else "No", # Arctic/Antarctic circles
        QUESTION_IDS["IN_NORTHERN_HEMISPHERE"]: "Yes" if lat > 0 else ("No" if lat < 0 else "Equator"),
        QUESTION_IDS["CONTINENT"]: continent,
        QUESTION_IDS["DRIVING_SIDE"]: "Left" if meta["country_code"] in LEFT_DRIVING_COUNTRY_CODES else "Right",
        QUESTION_IDS["COUNTRY"]: country,
        QUESTION_IDS["NEAR_COAST"]: city_specific_answers_row.get('Near Coast', "Unknown"),
        QUESTION_IDS["IS_ISLAND"]: city_specific_answers_row.get('Island', "Unknown"),
        QUESTION_IDS["IS_DESERT_REGION"]: city_specific_answers_row.get('Desert Region', "Unknown"),
        QUESTION_IDS["IS_MOUNTAINOUS"]: city_specific_answers_row.get('Mountainous', "Unknown"),
        QUESTION_IDS["CLIMATE_TYPE"]: city_specific_answers_row.get('Climate Type', "Unknown"),
        QUESTION_IDS["IS_BIG_CITY"]: city_specific_answers_row.get('Big City', "Unknown"),
        QUESTION_IDS["IS_SMALL_TOWN"]: city_specific_answers_row.get('Small Town', "Unknown"),
        QUESTION_IDS["LANGUAGES"]: city_specific_answers_row.get('Languages', "Unknown"),
        QUESTION_IDS["STATE_PROVINCE"]: meta["state"],
        QUESTION_IDS["CITY"]: city,
        QUESTION_IDS["COORDINATES"]: f"{lat:.4f}, {lon:.4f}"
    }
    return answers

def parse_class_mapping(mapping_str: Optional[str]) -> dict:
    """Parses a JSON string map of class indices to area into a set of dict."""
    try:
        if isinstance(mapping_str, str):
            loaded_dict = json.loads(mapping_str)
            if isinstance(loaded_dict, dict):
                return loaded_dict 
        return {}
    except json.JSONDecodeError:
        return {}

def main():
    """Main function to process samples and generate answers."""
    parser = argparse.ArgumentParser(description="Generate auto-answers for image location data.")
    parser.add_argument("test_samples_csv", type=Path, help="Path to the test_samples.csv file.")
    parser.add_argument("per_city_answers_csv", type=Path, help="Path to the per_city_answers.csv file.")
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("test_samples_with_answers.csv"),
        help="Path for the output CSV file with generated answers."
    )
    args = parser.parse_args()

    if not args.test_samples_csv.is_file():
        print(f"Error: Test samples file not found at {args.test_samples_csv}")
        return
    if not args.per_city_answers_csv.is_file():
        print(f"Error: Per city answers file not found at {args.per_city_answers_csv}")
        return

    try:
        test_samples_df = pd.read_csv(args.test_samples_csv)
        per_city_answers_df = pd.read_csv(args.per_city_answers_csv)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    all_generated_answers = []

    print(f"Processing {len(test_samples_df)} samples...")
    for index, row in test_samples_df.iterrows():
        lat = row.get('lat')
        lon = row.get('lon')
        class_mapping_str = row.get('class_mapping') 

        if pd.isna(lat) or pd.isna(lon):
            print(f"Warning: Skipping row {index+1} due to missing lat/lon.")
            # Append empty answers or NaNs for this row if needed, or skip
            empty_answers = {f"answer_{qid}": pd.NA for qid in QUESTION_IDS.values()}
            all_generated_answers.append(empty_answers)
            continue

        image_classes = parse_class_mapping(class_mapping_str)
        
        generated_answers = auto_answer_questions(float(lat), float(lon), per_city_answers_df, image_classes)
        
        # Prefix answer keys for DataFrame columns
        prefixed_answers = {f"answer_{k}": v for k, v in generated_answers.items()}
        all_generated_answers.append(prefixed_answers)

    answers_df = pd.DataFrame(all_generated_answers)

    # Concatenate original DataFrame with the new answers DataFrame
    output_df = pd.concat([test_samples_df, answers_df], axis=1)

    try:
        output_df.to_csv(args.output_csv, index=False)
        print(f"Successfully processed samples. Output saved to: {args.output_csv}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main()