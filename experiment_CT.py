import torch
import os
import numpy as np
import argparse

# Import user-defined utilities
from utils.prepare_data import create_images_dataset, ReconstructionDataset
from utils.experiment import run_experiment

# --- Script Configuration ---
DEFAULT_IMAGES_DIR = '/local/s4283341/cito/CT_dataset'
DEFAULT_PROCESSED_DATA_DIR = "/local/s4283341/cito/processed_image_datasets"
N_VIEWS_LIST = [1024, 512, 256]
NOISE_LEVEL = 0.05
K_SPLITS_NOISE2INVERSE = 4
trained_models_base_dir = "/local/s4283341/cito/trained_models_images"

np.random.seed(42)
torch.manual_seed(42)

def load_images_dataset(
    n_views_arg: int,
    image_size_n_arg: int, # For resizing images to n x n
    images_base_dir_arg: str,
    processed_data_dir_arg: str
):
    print(f"Preparing image dataset for Views: {n_views_arg}, Size: {image_size_n_arg}x{image_size_n_arg}")

    os.makedirs(processed_data_dir_arg, exist_ok=True)
    
    dataset_filename_parts = [
        f"images_nviews{n_views_arg}",
        f"size{image_size_n_arg}",
        f"noise{NOISE_LEVEL}",
        f"k{K_SPLITS_NOISE2INVERSE}.pt"
    ]
    dataset_filename = "_".join(dataset_filename_parts)
    dataset_path = os.path.join(processed_data_dir_arg, dataset_filename)

    full_dataset_list = None
    if os.path.exists(dataset_path):
        print(f"Loading existing processed image dataset: {dataset_path}")
        full_dataset_list = torch.load(dataset_path, weights_only=False) # Set weights_only based on content
        if not isinstance(full_dataset_list, list) or (not full_dataset_list and os.path.getsize(dataset_path) > 0): # check if empty list but file not empty
                print(f"Loaded dataset from {dataset_path} is invalid (not a list or empty). Regenerating.")
                full_dataset_list = None
        elif not full_dataset_list: # Genuinely empty list from an empty file or intentional save
                print(f"Loaded dataset from {dataset_path} is empty. Considering regeneration if source images exist.")
            
    if full_dataset_list is None:
        print(f"Generating processed image dataset from: {images_base_dir_arg}")
        full_dataset_list = create_images_dataset(
            image_folder_path=images_base_dir_arg,
            target_size_n=image_size_n_arg,
            n_views=n_views_arg,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )

        if not full_dataset_list: # Handles case where create_images_dataset returns empty or None
            print(f"Failed to generate dataset from {images_base_dir_arg} for {n_views_arg} views, size {image_size_n_arg}. Cannot proceed.")
            return None # Explicitly return None
        
        print(f"Saving generated image dataset to: {dataset_path}")
        try:
            torch.save(full_dataset_list, dataset_path)
        except Exception as e:
            print(f"Error saving dataset to {dataset_path}: {e}.")

    if full_dataset_list is None: # Double check after potential generation/saving issues
        print(f"Image dataset could not be loaded or generated for {n_views_arg} views, size {image_size_n_arg}. Aborting.")
        return None
    
    print(f"Image dataset for {n_views_arg} views, size {image_size_n_arg} prepared successfully. Samples: {len(full_dataset_list)}")
    return full_dataset_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run CT Image Reconstruction Experiment on Image Datasets.")
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        required=True,
        choices=["UNet", "DnCNN", "REDNet"], # Assuming these are still the relevant choices
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

    dataset = load_images_dataset(
        n_views_arg=args.n_views,
        image_size_n_arg=args.image_size,
        images_base_dir_arg=args.images_dir,
        processed_data_dir_arg=args.processed_data_dir
    )

    if dataset:
        print(f"Proceeding to run experiment with {len(dataset)} samples.")
        run_experiment(
            model_type_arg=args.model_type, 
            n_views_arg=args.n_views,
            full_dataset_list=dataset,
            trained_models_base_dir=args.trained_models_base_dir
        )
    else:
        print(f"Could not load or generate dataset for {args.n_views} views, size {args.image_size} from {args.images_dir}. Exiting.")