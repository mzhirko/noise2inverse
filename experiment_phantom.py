import torch
import os
import numpy as np
import argparse
from utils.prepare_data import create_phantom_dataset
from utils.experiment import run_experiment

# --- Experiment Configuration ---
N_VIEWS_LIST = [1024, 512, 256]
TOTAL_PHANTOMS_PER_VIEW_SETTING = 1000
NOISE_LEVEL = 0.05
K_SPLITS_NOISE2INVERSE = 4
PHANTOM_DATASET_DIR = "./phantom_datasets"

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def parse_args():
    """Parses command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Run CT Image Reconstruction Experiment.")
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
        help=f"Number of views for the dataset. Recommended from: {N_VIEWS_LIST}."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=PHANTOM_DATASET_DIR,
        help="Directory to load/save phantom datasets."
    )
    args = parser.parse_args()
    return args

def load_phantom_dataset(n_views_arg: int, phantom_dataset_dir: str):
    """Loads a pre-generated phantom dataset or creates it if it doesn't exist."""
    print(f"Preparing dataset for Views: {n_views_arg}")
    os.makedirs(phantom_dataset_dir, exist_ok=True)
    
    # Define a unique filename for the dataset based on its parameters.
    dataset_filename = (
        f"phantoms_nviews{n_views_arg}_total{TOTAL_PHANTOMS_PER_VIEW_SETTING}"
        f"_noise{NOISE_LEVEL}_k{K_SPLITS_NOISE2INVERSE}.pt"
    )
    dataset_path = os.path.join(phantom_dataset_dir, dataset_filename)

    full_dataset_list = None
    # If a pre-generated dataset file exists, try to load it.
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset: {dataset_path}")
        try:
            full_dataset_list = torch.load(dataset_path, weights_only=False)
            # If the loaded file is invalid or empty, set to None to trigger regeneration.
            if not isinstance(full_dataset_list, list) or not full_dataset_list:
                print(f"Loaded dataset from {dataset_path} is invalid or empty. Regenerating.")
                full_dataset_list = None
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}. Regenerating.")
            full_dataset_list = None
            
    # If the dataset wasn't loaded, generate a new one.
    if full_dataset_list is None:
        print(f"Generating {TOTAL_PHANTOMS_PER_VIEW_SETTING} phantoms for {n_views_arg} views...")
        full_dataset_list = create_phantom_dataset(
            n=TOTAL_PHANTOMS_PER_VIEW_SETTING,
            n_views=n_views_arg,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )

        # If generation is successful, save the dataset to a file for future runs.
        if full_dataset_list:
            print(f"Saving generated dataset to: {dataset_path}")
            torch.save(full_dataset_list, dataset_path)
        else:
            print(f"Failed to generate dataset for {n_views_arg} views.")
            return None

    if not full_dataset_list:
        print(f"Dataset could not be loaded or generated for {n_views_arg} views. Aborting.")
        return None
        
    print(f"Dataset for {n_views_arg} views prepared successfully.")
    return full_dataset_list

if __name__ == '__main__':
    args = parse_args()

    # Load or generate the dataset for the specified number of views.
    dataset = load_phantom_dataset(n_views_arg=args.n_views, phantom_dataset_dir=args.dataset_dir)

    # If the dataset is available, run the training experiment.
    if dataset:
        run_experiment(
            model_type_arg=args.model_type, 
            n_views_arg=args.n_views, 
            full_dataset_list=dataset
        )
    else:
        print(f"Could not load or generate dataset for {args.n_views} views. Exiting.")