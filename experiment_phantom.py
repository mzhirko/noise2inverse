import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import csv
import argparse

# Assuming these utils are in a 'utils' directory relative to this script
from utils.models import UNet, DnCNN, REDNet
from utils.prepare_data import create_phantom_dataset, ReconstructionDataset
from utils.train import train_model, DEVICE

# --- Experiment Configuration ---
N_VIEWS_LIST = [1024, 512, 256, 128]  # Still useful for choices and batch script
TOTAL_PHANTOMS_PER_VIEW_SETTING = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VAL_RATIO

NOISE_LEVEL = 0.05
K_SPLITS_NOISE2INVERSE = 4  # Used in create_phantom_dataset

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 16

PHANTOM_DATASET_DIR = "/local/s4283341/cito/phantom_datasets"
TRAINED_MODELS_BASE_DIR = "/local/s4283341/cito/trained_models"

def run_experiment(model_type_arg: str, n_views_arg: int):
    print(f"Starting experiment for Model: {model_type_arg}, Views: {n_views_arg} on DEVICE: {DEVICE}")

    # --- Dataset Preparation ---
    os.makedirs(PHANTOM_DATASET_DIR, exist_ok=True)
    dataset_filename = (
        f"phantoms_nviews{n_views_arg}_total{TOTAL_PHANTOMS_PER_VIEW_SETTING}"
        f"_noise{NOISE_LEVEL}_k{K_SPLITS_NOISE2INVERSE}.pt"
    )
    dataset_path = os.path.join(PHANTOM_DATASET_DIR, dataset_filename)

    full_dataset_list = None
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset: {dataset_path}")
        try:
            full_dataset_list = torch.load(dataset_path)
            if not isinstance(full_dataset_list, list) or not full_dataset_list:
                print(f"Loaded dataset from {dataset_path} is invalid or empty. Regenerating.")
                full_dataset_list = None # Force regeneration
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}. Regenerating.")
            full_dataset_list = None # Force regeneration
            
    if full_dataset_list is None:
        print(f"Generating {TOTAL_PHANTOMS_PER_VIEW_SETTING} phantoms for {n_views_arg} views...")
        full_dataset_list = create_phantom_dataset(
            n=TOTAL_PHANTOMS_PER_VIEW_SETTING,
            n_views=n_views_arg,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )

        if not full_dataset_list:
            print(f"Failed to generate dataset for {n_views_arg} views. Skipping this experiment run.")
            return
        
        print(f"Saving generated dataset to: {dataset_path}")
        try:
            torch.save(full_dataset_list, dataset_path)
        except Exception as e:
            print(f"Error saving dataset to {dataset_path}: {e}.")
            # Decide if you want to proceed with the in-memory dataset or stop
            # For now, we'll proceed if saving fails but generation was successful.
            # If saving is critical, you might want to return here.

    num_total_samples = len(full_dataset_list)
    num_train = int(TRAIN_RATIO * num_total_samples)
    num_val = int(VAL_RATIO * num_total_samples)
    
    indices = np.random.permutation(num_total_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    # test_indices = indices[num_train + num_val :] # Available if needed later

    train_data_list = [full_dataset_list[i] for i in train_indices]
    val_data_list = [full_dataset_list[i] for i in val_indices]
    
    train_dataset = ReconstructionDataset(train_data_list)
    val_dataset = ReconstructionDataset(val_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model Selection and Training ---
    model_generators = {
        "UNet": lambda: UNet(in_channels=1, out_channels=1),
        "DnCNN": lambda: DnCNN(in_channels=1, out_channels=1),
        "REDNet": lambda: REDNet(in_channels=1, out_channels=1)
    }

    if model_type_arg not in model_generators:
        print(f"Error: Model type '{model_type_arg}' is not recognized. Choose from {list(model_generators.keys())}.")
        return

    print(f"\nTraining {model_type_arg} for {n_views_arg} views...")
    
    current_model = model_generators[model_type_arg]()
    current_model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
    
    # Define save directory for this specific model and view setting
    save_directory = os.path.join(TRAINED_MODELS_BASE_DIR, f"views_{n_views_arg}", model_type_arg)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    
    # Model will be saved as: trained_models/views_{n_views}/{model_type}/{model_type}_views_{n_views}.pth
    model_save_name_stem = f"{model_type_arg}_views_{n_views_arg}" 

    epoch_losses, epoch_psnrs, epoch_ssims = train_model(
        model=current_model,
        dataloader=train_loader, # Should be train_loader, val_loader can be passed if train_model supports it for validation
        val_dataloader=val_loader, # Assuming train_model can take val_loader
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        model_name=model_save_name_stem, # train_model will append .pth
        save_path=save_directory # Directory where model is saved
    )
    print(f"Finished training {model_type_arg} for {n_views_arg} views.")
    print(f"Model saved in directory: {save_directory}")

    # --- Save Training History ---
    # CSV will be saved as: trained_models/views_{n_views}/{model_type}/{model_type}_views_{n_views}_training_history.csv
    history_csv_filename = f"{model_save_name_stem}_training_history.csv"
    history_csv_path = os.path.join(save_directory, history_csv_filename)
    
    try:
        with open(history_csv_path, 'w', newline='') as csvfile:
            history_writer = csv.writer(csvfile)
            history_writer.writerow(['Epoch', 'Loss', 'PSNR', 'SSIM']) # Assuming these are training metrics
            # If train_model returns validation metrics as well, you might want to log them
            for epoch_num in range(len(epoch_losses)): # Use len(epoch_losses) in case of early stopping
                history_writer.writerow([
                    epoch_num + 1, 
                    epoch_losses[epoch_num], 
                    epoch_psnrs[epoch_num] if epoch_num < len(epoch_psnrs) else 'N/A', 
                    epoch_ssims[epoch_num] if epoch_num < len(epoch_ssims) else 'N/A'
                ])
        print(f"Training history saved to {history_csv_path}")
    except Exception as e:
        print(f"Error saving training history to {history_csv_path}: {e}")

    print(f"\n--- Experiment run for Model: {model_type_arg}, Views: {n_views_arg} Finished ---")

if __name__ == '__main__':
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
    args = parser.parse_args()

    if args.n_views not in N_VIEWS_LIST:
        print(f"Warning: {args.n_views} is not in the predefined N_VIEWS_LIST {N_VIEWS_LIST}.")
        # Allow proceeding, but warn the user.
        # Alternatively, make it an error:
        # print(f"Error: Invalid number of views. Choose from {N_VIEWS_LIST}.")
        # exit(1)

    run_experiment(model_type_arg=args.model_type, n_views_arg=args.n_views)
