import os
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Your Cropping Function (as provided) ---
def crop_bottom_rows(image_path: Union[str, Path], num_rows_to_crop: int) -> Optional[Image.Image]:
    """
    Reads an image from a file path and crops the bottom few rows.

    Args:
        image_path: The path to the image file.
        num_rows_to_crop: The number of rows to crop from the bottom.

    Returns:
        The cropped image object (Pillow.Image.Image), or None if an error occurs.
    """
    try:
        img = Image.open(image_path).convert("RGB") # Ensure image is in RGB
        width, height = img.size

        if not isinstance(num_rows_to_crop, int) or num_rows_to_crop < 0:
            # print(f"Error: Number of rows to crop ({num_rows_to_crop}) must be a non-negative integer for {image_path}.")
            return None
        if num_rows_to_crop >= height:
            # print(f"Error: Number of rows to crop ({num_rows_to_crop}) is >= image height ({height}) for {image_path}. Cropping would result in an empty or invalid image.")
            return None

        left, upper, right, lower = 0, 0, width, height - num_rows_to_crop
        
        if lower <= upper: # Crop results in zero or negative height
            # print(f"Warning: Calculated crop area for {image_path} results in zero or negative height (lower: {lower}, upper: {upper}). Returning None.")
            return None
            
        cropped_img = img.crop((left, upper, right, lower))
        return cropped_img
    except FileNotFoundError:
        # print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        # print(f"An unexpected error occurred while cropping {image_path}: {e}")
        return None

# --- PyTorch Dataset Class ---
class GeochainPyTorchDataset(Dataset):
    """
    PyTorch Dataset for the Geochain data, using test_samples.csv.
    Reads images, crops them, and returns the cropped image along with metadata.
    """
    def __init__(self, 
                 csv_file_path: Union[str, Path], 
                 original_images_base_dir: Union[str, Path], 
                 num_rows_to_crop: int,
                 image_transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None): # For labels/targets if any
        """
        Args:
            csv_file_path (Union[str, Path]): Path to the test_samples.csv file.
            original_images_base_dir (Union[str, Path]): Base directory where original images 
                                                         (e.g., 'mapillary_dataset/') are stored.
            num_rows_to_crop (int): Number of rows to crop from the bottom of each image.
            image_transform (Optional[Any]): torchvision transforms to be applied to the cropped image.
            target_transform (Optional[Any]): torchvision transforms for labels (if you add them).
        """
        try:
            self.data_frame = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at {csv_file_path}")
        
        self.original_images_base_dir = Path(original_images_base_dir)
        self.num_rows_to_crop = num_rows_to_crop
        self.image_transform = image_transform
        self.target_transform = target_transform

        # Expected columns from test_samples.csv:
        # key, locatability_score, lat, lon, city, sub_folder, class_mapping, sequence_key
        # We will use 'key', 'city', 'sub_folder' to construct the image path.
        # We will return the cropped image and other relevant metadata.

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]

        image_key = str(row["key"])
        city_for_path = str(row["city"])
        sub_folder_for_path = str(row["sub_folder"])

        # Construct the path to the original image
        # Assumes original images are in: original_images_base_dir/city/sub_folder/images/key.jpg
        original_image_path = self.original_images_base_dir / \
                              city_for_path / \
                              sub_folder_for_path / \
                              "images" / \
                              f"{image_key}.jpg" # Assuming .jpg, adjust if necessary

        # Load and crop the image
        cropped_image_pil = crop_bottom_rows(original_image_path, self.num_rows_to_crop)

        if cropped_image_pil is None:
            # Handle cases where image loading or cropping fails
            # Option 1: Return a placeholder / skip (might require custom collate_fn in DataLoader)
            # Option 2: Raise an error
            # Option 3: Return a default tensor/image
            print(f"Warning: Could not load or crop image for key {image_key} at path {original_image_path}. Returning None for image.")
            # For simplicity, we'll return None here, but you might want a more robust handling.
            # If returning None, your training loop or collate_fn needs to handle it.
            # A common practice is to transform even a placeholder to a tensor.
            # For this example, we'll create a dummy tensor if image is None AFTER potential transform.
            image_tensor = torch.zeros((3, 64, 64)) # Example dummy tensor C, H, W
            if self.image_transform and cropped_image_pil is not None: # Check cropped_image_pil again
                 image_tensor = self.image_transform(cropped_image_pil)
            elif self.image_transform and cropped_image_pil is None: # Transform a dummy PIL if needed
                 placeholder_pil = Image.new('RGB', (64,64), color='grey')
                 image_tensor = self.image_transform(placeholder_pil) # Transform placeholder
        else: # Image successfully loaded and cropped
             if self.image_transform:
                image_tensor = self.image_transform(cropped_image_pil)
             else: # If no transform, convert PIL to Tensor manually (basic)
                # This is a basic conversion, torchvision.transforms.ToTensor() is better
                # as it scales to [0,1] and handles channel order.
                # For raw PIL, ensure your model can handle it or add ToTensor to transforms.
                # For now, let's assume image_transform will include ToTensor()
                raise ValueError("image_transform should include ToTensor() or similar.")


        # Extract other relevant data, converting dtypes as necessary
        # Ensure these column names match your test_samples.csv exactly
        locatability_score = float(row.get("locatability_score", float('nan')))
        lat = float(row.get("lat", float('nan')))
        lon = float(row.get("lon", float('nan')))
        class_mapping = str(row.get("class_mapping", "")) # Or int if it's always an integer
        sequence_key = str(row.get("sequence_key", ""))

        sample = {
            'image': image_tensor,
            'locatability_score': torch.tensor(locatability_score, dtype=torch.float32),
            'latitude': torch.tensor(lat, dtype=torch.float32),
            'longitude': torch.tensor(lon, dtype=torch.float32),
            'class_mapping': class_mapping, # Keep as string or convert to tensor if categorical
            'sequence_key': sequence_key,
            'original_image_key': image_key # Good for debugging
        }
        
        # Example: if class_mapping is a label you want to transform to a tensor
        if self.target_transform and 'class_mapping_tensor' in self.target_transform: # Fictional check
             sample['class_mapping_tensor'] = self.target_transform(class_mapping)


        return sample

# --- Example Usage ---
if __name__ == '__main__':
    # --- Configuration - Adjust these paths and parameters ---
    PATH_TO_CSV = "data/test_samples.csv"  # Path to your test_samples.csv
    BASE_IMAGES_DIR = "mapillary" # Path to the FOLDER CONTAINING 'city' subfolders, etc.
                                          # e.g., if images are in 'mapillary_dataset/london/a/images/1.jpg'
                                          # then BASE_IMAGES_DIR is 'mapillary_dataset'
    NUM_ROWS_TO_CROP = 30             # Number of rows to crop from the bottom
    BATCH_SIZE = 4                    # Batch size for DataLoader
    IMAGE_HEIGHT = 224                # Example image height for resizing
    IMAGE_WIDTH = 224                 # Example image width for resizing

    # Define standard image transformations for PyTorch models
    # (Resize, ToTensor, Normalize - adapt as needed for your model)
    image_transforms = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(), # Converts PIL image (H x W x C) in range [0, 255] to
                              # a FloatTensor (C x H x W) in range [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # 1. Create an instance of your custom Dataset
    print(f"Loading dataset from CSV: {PATH_TO_CSV}")
    print(f"Base directory for original images: {BASE_IMAGES_DIR}")
    
    # Check if CSV and base image directory exist before creating dataset
    if not os.path.exists(PATH_TO_CSV):
        print(f"ERROR: CSV file not found at '{PATH_TO_CSV}'. Please check the path.")
        exit()
    if not os.path.isdir(BASE_IMAGES_DIR):
        print(f"ERROR: Base images directory not found at '{BASE_IMAGES_DIR}'. Please check the path.")
        exit()

    geochain_dataset = GeochainPyTorchDataset(
        csv_file_path=PATH_TO_CSV,
        original_images_base_dir=BASE_IMAGES_DIR,
        num_rows_to_crop=NUM_ROWS_TO_CROP,
        image_transform=image_transforms
    )

    print(f"Dataset created. Number of samples: {len(geochain_dataset)}")

    if len(geochain_dataset) == 0:
        print("Dataset is empty. Please check your CSV file or image paths.")
        exit()

    # 2. Create a DataLoader
    data_loader = DataLoader(
        geochain_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle data for training, set to False for validation/testing
        num_workers=0   # Start with 0 for debugging, increase later (e.g., 2, 4)
                        # On Windows, num_workers > 0 can sometimes cause issues if not careful with __main__
    )

    print(f"DataLoader created. Batch size: {BATCH_SIZE}")

    # 3. Iterate through the DataLoader (example of how to get a batch)
    print("\nFetching a sample batch from DataLoader...")
    try:
        for i, batch in enumerate(data_loader):
            print(f"\n--- Batch {i+1} ---")
            images = batch['image']
            loc_scores = batch['locatability_score']

            print(f"Images batch shape: {images.shape}") # Should be (BATCH_SIZE, C, H, W)
            print(f"Locatability scores batch shape: {loc_scores.shape}")
            print(f"Sequence key of first item in batch: {batch['sequence_key'][0]}") 
                
    except Exception as e:
        print(f"\nAn error occurred while iterating through DataLoader: {e}")
        print("This might be due to issues with image paths, file reading, or data processing.")
        print("Make sure all image files listed in the CSV exist at the expected locations and are readable.")

    print("\nExample finished.")