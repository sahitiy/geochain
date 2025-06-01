🗺️ GeoChain Dataset
This repository hosts the GeoChain Benchmark, a large-scale benchmark introduced in the paper "GeoChain: Multimodal Chain-of-Thought for Geographic Reasoning" for evaluating step-by-step geographic reasoning in multimodal large language models (MLLMs).

Leveraging 1.46 million Mapillary street-level images, GeoChain pairs each image with a 21-step chain-of-thought (CoT) question sequence, resulting in over 30 million Q&A pairs. These sequences are designed to guide models from coarse attributes to fine-grained localization, covering four key reasoning categories: visual, spatial, cultural, and precise geolocation, with annotations for difficulty. Images within the dataset are also enriched with semantic segmentation (150 classes) and a visual locatability score.

Our benchmarking of contemporary MLLMs (including GPT-4.1 variants, Claude 3.7, and Gemini 2.5 variants) on a diverse 2,088-image subset reveals consistent challenges: models frequently exhibit weaknesses in visual grounding, display erratic reasoning, and struggle to achieve accurate localization, especially as reasoning complexity escalates. GeoChain offers a robust diagnostic methodology, critical for fostering significant advancements in complex geographic reasoning within MLLMs.

📦 Dataset Access & Overview
This Git repository primarily contains the code for the GeoChain benchmark and a smaller test_samples.csv file (located in the data/ directory). 

For the complete GeoChain dataset, including main_test_samples.csv, please visit our Hugging Face Datasets repository:

➡️ https://huggingface.co/datasets/sahitiy51/geochain

The Hugging Face repository provides easy access to load the data directly using the datasets library.

💻 Code & Usage
The code/ directory in this repository contains scripts related to dataset generation, processing, and evaluation:

compute_scores_mapillary.py: Computes visual locatability scores for images.
generate_dataset_split.py: Creates the main_test_samples.csv and test_samples.csv splits from processed Mapillary data.
generate_answers.py: Merges test samples with ground truth answers.
geochain_pytorch_dataset.py: Provides a PyTorch Dataset class for convenient loading and preprocessing of data (like test_samples.csv). It handles reading the CSV, constructing image paths, loading original images, applying the defined cropping logic, and preparing samples for PyTorch DataLoaders.
requirements.txt: Lists the Python dependencies for running these scripts.


Environment Setup
To prepare your environment to use the provided Python scripts:

Install Dependencies:

pip install -r code/requirements.txt

Generating the Dataset:

1. Download Mapillary Dataset:
Download the original "Mapillary Street-level Sequences Dataset (Vistas)" from its official source (e.g., https://www.mapillary.com/dataset/places) and extract it to a chosen folder.

2. Compute Locatability Scores:

python code/compute_scores_mapillary.py --root_folder /path/to/your/mapillary_image_data

3. Create Dataset Splits:

python code/generate_dataset_split.py --root_folder /path/to/your/mapillary_image_data
This will generate test_samples.csv and main_test_samples.csv in the data/ directory.

4. Generate an Answer File:
This script merges test samples with the ground truth answers.


python code/generate_answers.py \
    data/test_samples.csv \
    data/per_city_answers.csv \
    --output_csv data/test_samples_with_answers.csv

📜 License
This dataset and accompanying code are distributed under the same license as the Mapillary Street Level Sequences Dataset. Please review Mapillary’s licensing terms before use.

