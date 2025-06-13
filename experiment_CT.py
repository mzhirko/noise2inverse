import torch
import os
import numpy as np
import argparse

from utils.prepare_data import create_images_dataset, ReconstructionDataset
from utils.experiment import run_experiment

# --- Script Configuration ---
DEFAULT_IMAGES_DIR = './CT_dataset'
DEFAULT_PROCESSED_DATA_DIR = "./processed_image_datasets"
N_VIEWS_LIST = [1024, 512, 256]
NOISE_LEVEL = 0.05
K_SPLITS_NOISE2INVERSE = 4
trained_models_base_dir = "./trained_models_images"

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    """Parses command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Run CT Image Reconstruction Experiment on Image Datasets.")
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        required=True,
        choices=["UNet", "DnCNN", "REDNet"],
        help="Type of model to train (UNet, DnCNN, REDNet)."
    )
    parser.add_argument(
        "-v", "--n_views",
        type=int,
        required=True,
        help=f"Number of views for sinogram generation. Recommended from: {N_VIEWS_LIST}."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=True,
        help="Target size 'n' to resize images to (n x n)."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help=f"Directory containing the input images. Default: {DEFAULT_IMAGES_DIR}"
    )
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=DEFAULT_PROCESSED_DATA_DIR,
        help=f"Directory to load/save processed image datasets (.pt files). Default: {DEFAULT_PROCESSED_DATA_DIR}"
    )
    parser.add_argument(
        "--trained_models_base_dir",
        type=str,
        default=trained_models_base_dir,
        help=f"Directory to save trained_models and results Default: {trained_models_base_dir}"
    )
    
    args = parser.parse_args()
    return args

def load_images_dataset(
    n_views_arg: int,
    image_size_n_arg: int,
    images_base_dir_arg: str,
    processed_data_dir_arg: str
):
    """Loads a pre-processed dataset or creates it if it doesn't exist."""
    print(f"Preparing image dataset for Views: {n_views_arg}, Size: {image_size_n_arg}x{image_size_n_arg}")

    os.makedirs(processed_data_dir_arg, exist_ok=True)
    
    # Construct a unique filename for the dataset based on its parameters.
    dataset_filename_parts = [
        f"images_nviews{n_views_arg}",
        f"size{image_size_n_arg}",
        f"noise{NOISE_LEVEL}",
        f"k{K_SPLITS_NOISE2INVERSE}.pt"
    ]
    dataset_filename = "_".join(dataset_filename_parts)
    dataset_path = os.path.join(processed_data_dir_arg, dataset_filename)

    full_dataset_list = None
    # If a pre-processed dataset file exists, load it.
    if os.path.exists(dataset_path):
        print(f"Loading existing processed image dataset: {dataset_path}")
        full_dataset_list = torch.load(dataset_path, weights_only=False)
        # Validate the loaded data; if it's not a list or seems invalid, trigger regeneration.
        if not isinstance(full_dataset_list, list) or (not full_dataset_list and os.path.getsize(dataset_path) > 0):
                print(f"Loaded dataset from {dataset_path} is invalid. Regenerating.")
                full_dataset_list = None
        elif not full_dataset_list:
                print(f"Loaded dataset from {dataset_path} is empty.")
            
    # If the dataset was not loaded, generate it from the source images.
    if full_dataset_list is None:
        print(f"Generating processed image dataset from: {images_base_dir_arg}")
        full_dataset_list = create_images_dataset(
            image_folder_path=images_base_dir_arg,
            target_size_n=image_size_n_arg,
            n_views=n_views_arg,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )
        # If dataset creation is successful, save it to a file for future use.
        if full_dataset_list:
            print(f"Saving generated image dataset to: {dataset_path}")
            torch.save(full_dataset_list, dataset_path)
        else:
            print(f"Failed to generate dataset from {images_base_dir_arg}.")
            return None

    if full_dataset_list is None:
        print(f"Image dataset could not be loaded or generated. Aborting.")
        return None
    
    print(f"Image dataset prepared successfully. Samples: {len(full_dataset_list)}")
    return full_dataset_list

if __name__ == '__main__':
    args = parse_args()

    # Load or generate the required dataset.
    dataset = load_images_dataset(
        n_views_arg=args.n_views,
        image_size_n_arg=args.image_size,
        images_base_dir_arg=args.images_dir,
        processed_data_dir_arg=args.processed_data_dir
    )

    # If the dataset is ready, proceed with the training experiment.
    if dataset:
        print(f"Proceeding to run experiment with {len(dataset)} samples.")
        run_experiment(
            model_type_arg=args.model_type, 
            n_views_arg=args.n_views,
            full_dataset_list=dataset,
            trained_models_base_dir=args.trained_models_base_dir
        )
    else:
        print(f"Could not load or generate dataset. Exiting.")