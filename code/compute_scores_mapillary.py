import os
import torch
import pandas as pd
import argparse
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
from torch.utils.data import Dataset
from compute_scores_batch import compute_scores
import glob

class ImageSegmentationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_key = (image_path.split('/')[-1]).split('.')[0]
        try:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size[::-1] 
            return image, original_size, image_key
        except FileNotFoundError:
            print(f"Error: Image not found at {image_path} in Dataset. Returning placeholder.")
            return None, None, image_key 
        except Exception as e:
            print(f"Error loading image {image_path} in Dataset: {e}. Returning placeholder.")
            return None, None, image_key


def process_mapillary_data(root_folder_path):
    """
    Processes the Mapillary street sequences dataset structure.

    Args:
        root_folder_path (str): The path to the root folder containing city data.
    """
    if not os.path.isdir(root_folder_path):
        print(f"Error: Root folder '{root_folder_path}' not found.")
        return

    print(f"Processing data in root folder: {root_folder_path}\n")
    try:
        city_folders = [d for d in os.listdir(root_folder_path)
                        if os.path.isdir(os.path.join(root_folder_path, d))]
    except OSError as e:
        print(f"Error accessing root folder contents: {e}")
        return

    if not city_folders:
        print(f"No city folders found in '{root_folder_path}'.")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "facebook/mask2former-swin-large-ade-semantic"
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    except OSError:
        print(f"Could not load {model_id}. Trying 'facebook/maskformer-swin-large-ade-semantic'...")
        model_id = "facebook/maskformer-swin-large-ade-semantic"
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, use_fast=True)

    model.to(device)
    model.eval()


    for city_name in city_folders:
        city_path = os.path.join(root_folder_path, city_name)
        print(f"Processing city: {city_name}")

        subfolder_types = ["database", "query"]
        for subfolder_type in subfolder_types:
            subfolder_path = os.path.join(city_path, subfolder_type)
            print(f'Running eval for folder {subfolder_path}')
            image_paths = glob.glob(os.path.join(subfolder_path, 'images/*'))
            dataset = ImageSegmentationDataset(image_paths)
            output_df = compute_scores(model, processor, dataset)
            if os.path.exists(subfolder_path):
                output_df.to_csv(os.path.join(subfolder_path, 'locatability_scores.csv'))
        print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Mapillary dataset structure and read raw.csv files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--root_folder",
        type=str
    )

    args = parser.parse_args()
    process_mapillary_data(args.root_folder)