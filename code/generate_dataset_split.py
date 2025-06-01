import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd

# --- Constants ---
LOCATABILITY_SCORES_FILENAME = "locatability_scores.csv"
RAW_CSV_FILENAME = "raw.csv"
SEQ_INFO_FILENAME = "seq_info.csv"
DEFAULT_TARGET_COLUMNS = ['key', 'locatability_score', 'lat', 'lon', 'city', 'sub_folder', 'class_mapping', 'sequence_key']
DEFAULT_SUBFOLDER_TYPES = ['database', 'query']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate datasets based on locatability scores.")

    parser.add_argument("--root_folder", type=Path, required=True,
                        help="Root folder where city data is located.")
    parser.add_argument("--output_mini_test_file", type=Path, default=Path("mini_test_samples.csv"),
                        help="Output CSV file name for the mini test set.")
    parser.add_argument("--output_main_test_file", type=Path, default=Path("main_test_samples.csv"),
                        help="Output CSV file name for the main test set.")

    parser.add_argument("--easy_threshold_low", type=float, default=0.45,
                        help="Lower bound for 'easy' locatability score.")
    parser.add_argument("--easy_threshold_high", type=float, default=0.6,
                        help="Upper bound for 'easy' locatability score (exclusive).")
    parser.add_argument("--hard_threshold_low", type=float, default=0.12,
                        help="Lower bound for 'hard' locatability score.")
    parser.add_argument("--hard_threshold_high", type=float, default=0.22,
                        help="Upper bound for 'hard' locatability score (exclusive).")
    parser.add_argument("--num_sequences_per_category_test", type=int, default=700,
                        help="Number of unique sequences to sample for each category for the final test set.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    if not args.root_folder.is_dir():
        parser.error(f"Root folder not found: {args.root_folder}")

    # Ensure medium thresholds are derived correctly
    args.medium_threshold_low = args.hard_threshold_high
    args.medium_threshold_high = args.easy_threshold_low

    return args

def load_locatability_data_for_city(
    city_path: Path,
    subfolder_types: List[str]
) -> Dict[str, pd.DataFrame]:
    """Loads locatability data for a single city from its subfolders."""
    city_loc_data: Dict[str, pd.DataFrame] = {}
    for subfolder_type in subfolder_types:
        subfolder_path = city_path / subfolder_type
        locatability_file_path = subfolder_path / LOCATABILITY_SCORES_FILENAME
        if locatability_file_path.exists():
            try:
                city_loc_data[subfolder_type] = pd.read_csv(locatability_file_path)
            except Exception as e:
                logger.error(f"Error reading {locatability_file_path}: {e}")
        else:
            logger.warning(f"Locatability file not found: {locatability_file_path}")
    return city_loc_data

def load_all_locatability_data(
    root_folder: Path,
    subfolder_types: List[str] = DEFAULT_SUBFOLDER_TYPES
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Loads all locatability data from CSVs into a nested dictionary."""
    locatability_city_mapping: Dict[str, Dict[str, pd.DataFrame]] = {}
    logger.info(f"Loading locatability data from: {root_folder}")
    for city_folder_item in root_folder.iterdir():
        if city_folder_item.is_dir():
            city_name = city_folder_item.name
            logger.info(f"Processing city: {city_name}")
            city_data = load_locatability_data_for_city(city_folder_item, subfolder_types)
            if city_data:
                 locatability_city_mapping[city_name] = city_data
    logger.info("Finished loading locatability data.")
    return locatability_city_mapping

def pick_samples_in_score_range(
    df: pd.DataFrame,
    score_column_name: str,
    lower_bound: float,
    higher_bound: float,
    num_samples: int = -1,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Filters a DataFrame for rows where 'score_column_name' is within [lower_bound, higher_bound)
    and samples 'num_samples' rows. If num_samples is -1, all filtered rows are returned.
    """
    filtered_df = df[(df[score_column_name] >= lower_bound) & (df[score_column_name] < higher_bound)]

    if filtered_df.empty:
        logger.debug(f"Filtered DataFrame is empty for range [{lower_bound}, {higher_bound}) on column '{score_column_name}'.")
        return filtered_df

    if num_samples == -1 or len(filtered_df) <= num_samples:
        return filtered_df
    
    return filtered_df.sample(n=num_samples, random_state=random_seed)

def generate_dataset_for_category(
    root_folder: Path,
    locatability_city_mapping_data: Dict[str, Dict[str, pd.DataFrame]],
    lower_bound: float,
    higher_bound: float,
    target_columns: List[str] = DEFAULT_TARGET_COLUMNS,
    subfolder_types: List[str] = DEFAULT_SUBFOLDER_TYPES
) -> pd.DataFrame:
    """
    Generates a dataset for a specific locatability score range, gathering all relevant images.
    """
    all_category_samples_list = []

    for city, subfolder_data in locatability_city_mapping_data.items():
        for sub_folder_type, loc_df in subfolder_data.items():
            # Take all samples in the specified locatability range
            sampled_loc_rows = pick_samples_in_score_range(
                loc_df,
                'locatability_scores', # Column name in locatability_scores.csv
                lower_bound,
                higher_bound,
                num_samples=-1 # Take all matching
            )

            if sampled_loc_rows.empty:
                continue

            # Prepare initial output DataFrame from locatability data
            output_df = pd.DataFrame({
                'key': sampled_loc_rows['key'],
                'locatability_score': sampled_loc_rows['locatability_scores'], # Renaming
                'class_mapping': sampled_loc_rows['class_mapping']
            })

            city_subfolder_path = root_folder / city / sub_folder_type
            raw_csv_path = city_subfolder_path / RAW_CSV_FILENAME
            seq_info_csv_path = city_subfolder_path / SEQ_INFO_FILENAME

            if not (raw_csv_path.exists() and seq_info_csv_path.exists()):
                logger.warning(f"Missing {RAW_CSV_FILENAME} or {SEQ_INFO_FILENAME} for {city}/{sub_folder_type}. Skipping.")
                continue

            try:
                raw_df = pd.read_csv(raw_csv_path, usecols=['key', 'lat', 'lon'])
                sequence_df = pd.read_csv(seq_info_csv_path, usecols=['key', 'sequence_key'])
            except Exception as e:
                logger.error(f"Error reading CSVs for {city}/{sub_folder_type}: {e}")
                continue
            
            # Merge with raw data and sequence info
            merged_df = pd.merge(output_df, raw_df, on='key', how='inner')
            merged_df = pd.merge(merged_df, sequence_df, on='key', how='inner')

            merged_df['city'] = city
            merged_df['sub_folder'] = sub_folder_type
            
            # Ensure all target columns are present, fill with NA if not, and select/reorder
            for col in target_columns:
                if col not in merged_df.columns:
                    merged_df[col] = pd.NA
            
            all_category_samples_list.append(merged_df[target_columns])

    if not all_category_samples_list:
        return pd.DataFrame(columns=target_columns)
    
    return pd.concat(all_category_samples_list, ignore_index=True)

def sample_sequences_for_final_set(
    source_df: pd.DataFrame,
    num_sequences_to_sample: int,
    category_name: str,
    excluded_sequence_keys: Optional[Set[str]] = None,
    random_seed: Optional[int] = None
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Samples unique sequences for the final test set from a category DataFrame.
    Returns a DataFrame with one image per sampled sequence and the set of sampled sequence keys.
    """
    empty_df = pd.DataFrame(columns=source_df.columns if not source_df.empty else DEFAULT_TARGET_COLUMNS)
    
    if source_df.empty or 'sequence_key' not in source_df.columns:
        logger.warning(f"{category_name.capitalize()}_df is empty or missing 'sequence_key'. Cannot sample for final set.")
        return empty_df, set()

    df_to_sample_from = source_df.copy()
    if excluded_sequence_keys:
        df_to_sample_from = df_to_sample_from[~df_to_sample_from['sequence_key'].isin(list(excluded_sequence_keys))]

    unique_sequences = list(df_to_sample_from['sequence_key'].unique())
    if not unique_sequences:
        logger.warning(f"No unique {category_name} sequences available after filtering for final set.")
        return empty_df, set()

    actual_num_to_sample = min(num_sequences_to_sample, len(unique_sequences))
    if actual_num_to_sample < num_sequences_to_sample:
        logger.warning(
            f"Requested {num_sequences_to_sample} {category_name} sequences, but only "
            f"{len(unique_sequences)} available (after exclusions). Sampling {actual_num_to_sample}."
        )
    
    if actual_num_to_sample == 0:
        return empty_df, set()

    # Set seed for this specific random.sample call if random_seed is provided
    # Note: random.sample itself does not have a random_state. We seed globally.
    # If more fine-grained control for random.sample specifically is needed,
    # one might temporarily set random.seed() or use numpy.random.choice.
    # For simplicity, global seed is used via args.random_seed.
    sampled_sequence_keys_list = random.sample(unique_sequences, actual_num_to_sample)
    
    # Get all images for the sampled sequences
    df_with_sampled_sequences = df_to_sample_from[df_to_sample_from['sequence_key'].isin(sampled_sequence_keys_list)]

    # Sample one image per sequence
    if not df_with_sampled_sequences.empty:
        final_sampled_df = df_with_sampled_sequences.groupby('sequence_key', group_keys=False).sample(n=1, random_state=random_seed)
        return final_sampled_df, set(sampled_sequence_keys_list)
    
    return empty_df, set()


def main():
    """Main script execution."""
    args = parse_arguments()

    
    random.seed(args.random_seed)

    # 1. Load Locatability Data
    all_loc_data = load_all_locatability_data(args.root_folder)
    if not all_loc_data:
        logger.error("No locatability data loaded. Exiting.")
        return

    # 2. Generate Datasets for Each Category (Easy, Medium, Hard)
    logger.info("Generating initial easy dataset...")
    easy_df_initial = generate_dataset_for_category(
        args.root_folder, all_loc_data, args.easy_threshold_low, args.easy_threshold_high
    )
    logger.info(f"Generated initial easy_df with {len(easy_df_initial)} images.")

    logger.info("Generating initial medium dataset...")
    medium_df_initial = generate_dataset_for_category(
        args.root_folder, all_loc_data, args.medium_threshold_low, args.medium_threshold_high
    )
    logger.info(f"Generated initial medium_df with {len(medium_df_initial)} images.")

    logger.info("Generating initial hard dataset...")
    hard_df_initial = generate_dataset_for_category(
        args.root_folder, all_loc_data, args.hard_threshold_low, args.hard_threshold_high
    )
    logger.info(f"Generated initial hard_df with {len(hard_df_initial)} images.")

    # --- 3. Prepare Final Test Set ---
    logger.info("Preparing final test set...")
    
    # Sample Hard for Test Set
    final_hard_df, sampled_hard_seq_keys = sample_sequences_for_final_set(
        hard_df_initial, args.num_sequences_per_category_test, "hard", random_seed=args.random_seed
    )
    
    # Sample Medium for Test Set (excluding already sampled hard sequences)
    final_medium_df, sampled_medium_seq_keys = sample_sequences_for_final_set(
        medium_df_initial, args.num_sequences_per_category_test, "medium",
        excluded_sequence_keys=sampled_hard_seq_keys, random_seed=args.random_seed
    )
    
    # Sample Easy for Test Set (excluding already sampled hard and medium sequences)
    all_test_sampled_keys_so_far = sampled_hard_seq_keys.union(sampled_medium_seq_keys)
    final_easy_df, sampled_easy_seq_keys = sample_sequences_for_final_set(
        easy_df_initial, args.num_sequences_per_category_test, "easy",
        excluded_sequence_keys=all_test_sampled_keys_so_far, random_seed=args.random_seed
    )

    # Combine to form the final test DataFrame
    final_test_set_dfs = [df for df in [final_easy_df, final_medium_df, final_hard_df] if not df.empty]
    if final_test_set_dfs:
        final_test_df = pd.concat(final_test_set_dfs, ignore_index=True)
        logger.info(f"Final test set created with {len(final_test_df)} images from {len(final_test_df['sequence_key'].unique())} unique sequences.")
        logger.info(f"Test set head:\n{final_test_df.head()}")
        final_test_df.to_csv(args.output_mini_test_file, index=False)
        logger.info(f"Final test samples saved to {args.output_mini_test_file}")
    else:
        final_test_df = pd.DataFrame(columns=DEFAULT_TARGET_COLUMNS) # Ensure schema if empty
        logger.warning("Final test set is empty. Not saving.")
    logger.info("Finished preparing final test set.")

    # --- 4. Prepare Main Test Set ---
    # This set includes all images from the initial category DataFrames that are NOT from sequences used in the final test set.
    logger.info("Preparing full test set...")
    
    all_initial_categorized_dfs = [df for df in [easy_df_initial, medium_df_initial, hard_df_initial] if not df.empty]
    
    if all_initial_categorized_dfs:
        combined_initial_df = pd.concat(all_initial_categorized_dfs, ignore_index=True)
        
        # Get all sequence keys that are part of the final test set
        test_keys = set()
        if not final_test_df.empty and 'key' in final_test_df.columns:
            test_keys = set(final_test_df['key'].unique())

        if not combined_initial_df.empty and 'key' in combined_initial_df.columns:
            # Exclude images belonging to sequences used in the test set
            full_test_df = combined_initial_df[~combined_initial_df['key'].isin(list(test_keys))]
            
            if not full_test_df.empty:
                full_test_df.to_csv(args.output_main_test_file, index=False)
                logger.info(f"Full Test samples saved to {args.output_main_test_file} ({len(full_test_df)} images).")
            else:
                logger.warning("Full Test set is empty after excluding mini test set keys.")
        else:
            logger.warning("Initial combined DataFrame is empty or missing 'key'. Cannot create full test set.")
    else:
        logger.warning("No data available from initial category DataFrames to create a full test set.")
    
    logger.info("Script finished.")

if __name__ == "__main__":
    main()