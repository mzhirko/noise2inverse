import torch
import torch.nn as nn # Keep for type hinting or other nn components if needed elsewhere
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import csv
import random # Added for HPO sampling
import matplotlib.pyplot as plt # Added for saving images

# Import models and other utilities from the 'utils' directory
from utils.models import UNet, DnCNN, REDNet
from utils.prepare_data import create_phantom_dataset, ReconstructionDataset
from utils.train import train_model, DEVICE # Assumes DEVICE is defined in utils.train

# --- Experiment Configuration (Original values from your script) ---
TOTAL_PHANTOMS_PER_VIEW_SETTING = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VAL_RATIO

np.random.seed(42)
torch.manual_seed(42)
random.seed(42) # Seed for random sampling of validation image

NOISE_LEVEL = 0.01
K_SPLITS_NOISE2INVERSE = 4 # Used in dataset generation

NUM_EPOCHS = 30 # Original number of epochs for HPO runs

# Original directory paths from your script
PHANTOM_DATASET_DIR = "/local/s4283341/cito/phantom_datasets"
TRAINED_MODELS_BASE_DIR = "/local/s4283341/cito/trained_models"

# --- HPO Configuration ---
N_HPO_TRIALS = 20
FIXED_N_VIEWS_HPO = 360
FIXED_MODEL_TYPE_HPO = "DnCNN"


def run_single_hpo_trial(
    trial_id: int,
    model_type_arg: str,
    n_views_arg: int,
    lr_hpo: float,
    batch_size_hpo: int,
    num_layers_dncnn_hpo: int,
    num_features_dncnn_hpo: int
):
    print(f"--- HPO Trial {trial_id}/{N_HPO_TRIALS} ---")
    print(f"Config: Model={model_type_arg}, Views={n_views_arg}, LR={lr_hpo:.6f}, BS={batch_size_hpo}, Layers={num_layers_dncnn_hpo}, Feats={num_features_dncnn_hpo}")
    print(f"Device: {DEVICE}")

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
            # Assuming weights_only=False is correct for your dataset format
            full_dataset_list = torch.load(dataset_path, weights_only=False)
            if not isinstance(full_dataset_list, list) or not full_dataset_list:
                print(f"Loaded dataset from {dataset_path} is invalid or empty. Regenerating.")
                full_dataset_list = None
        except Exception as e:
            print(f"Error loading dataset from {dataset_path}: {e}. Regenerating.")
            full_dataset_list = None
            
    if full_dataset_list is None:
        print(f"Generating {TOTAL_PHANTOMS_PER_VIEW_SETTING} phantoms for {n_views_arg} views...")
        full_dataset_list = create_phantom_dataset(
            n=TOTAL_PHANTOMS_PER_VIEW_SETTING,
            n_views=n_views_arg,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )
        if not full_dataset_list:
            print(f"Failed to generate dataset for {n_views_arg} views. Skipping this trial.")
            return 0.0, 0.0 # Return neutral values for PSNR/SSIM
        
        print(f"Saving generated dataset to: {dataset_path}")
        try:
            torch.save(full_dataset_list, dataset_path)
        except Exception as e:
            print(f"Error saving dataset to {dataset_path}: {e}.")
            # Proceed with in-memory dataset if saving fails

    if full_dataset_list is None or not isinstance(full_dataset_list, list) or len(full_dataset_list) == 0:
        print(f"Dataset for {n_views_arg} views could not be loaded or generated. Aborting trial.")
        return 0.0, 0.0

    num_total_samples = len(full_dataset_list)
    num_train = int(TRAIN_RATIO * num_total_samples)
    num_val = int(VAL_RATIO * num_total_samples)
    
    if num_train == 0 or num_val == 0 or num_train + num_val > num_total_samples:
        print(f"Insufficient data for train/val split. Total: {num_total_samples}, Train: {num_train}, Val: {num_val}. Adjust ratios or dataset size.")
        return 0.0, 0.0

    indices = np.random.permutation(num_total_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    
    train_data_list = [full_dataset_list[i] for i in train_indices]
    val_data_list = [full_dataset_list[i] for i in val_indices]
    
    train_dataset = ReconstructionDataset(train_data_list)
    val_dataset = ReconstructionDataset(val_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size_hpo, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_hpo, shuffle=False, num_workers=0)

    # --- Model Selection and Training ---
    current_model = None
    if model_type_arg == "DnCNN":
        current_model = DnCNN(in_channels=1, out_channels=1, num_layers=num_layers_dncnn_hpo, num_features=num_features_dncnn_hpo)
    else:
        print(f"Error: Model type '{model_type_arg}' is not configured for this HPO setup.")
        return 0.0, 0.0

    current_model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(current_model.parameters(), lr=lr_hpo)
    
    hpo_trial_config_name = f"trial_{trial_id}_lr{lr_hpo:.1E}_bs{batch_size_hpo}_lay{num_layers_dncnn_hpo}_feat{num_features_dncnn_hpo}"
    save_directory = os.path.join(TRAINED_MODELS_BASE_DIR, f"views_{n_views_arg}", model_type_arg, "hpo_runs", hpo_trial_config_name)
    os.makedirs(save_directory, exist_ok=True)
    
    model_save_name_stem = f"{model_type_arg}_views_{n_views_arg}_trial_{trial_id}"

    epoch_losses, epoch_psnrs, epoch_ssims = train_model(
        model=current_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        model_name=model_save_name_stem,
        save_path=save_directory
    )
    
    final_val_psnr = epoch_psnrs[-1] if epoch_psnrs and len(epoch_psnrs) > 0 else 0.0
    final_val_ssim = epoch_ssims[-1] if epoch_ssims and len(epoch_ssims) > 0 else 0.0

    # --- Save Training History for this trial ---
    history_csv_filename = f"{model_save_name_stem}_training_history.csv"
    history_csv_path = os.path.join(save_directory, history_csv_filename)
    
    try:
        with open(history_csv_path, 'w', newline='') as csvfile:
            history_writer = csv.writer(csvfile)
            history_writer.writerow(['Epoch', 'Train_Loss', 'Val_PSNR', 'Val_SSIM'])
            for epoch_num in range(NUM_EPOCHS):
                train_loss = epoch_losses[epoch_num] if epoch_num < len(epoch_losses) else 'N/A'
                val_psnr_epoch = epoch_psnrs[epoch_num] if epoch_num < len(epoch_psnrs) else 'N/A'
                val_ssim_epoch = epoch_ssims[epoch_num] if epoch_num < len(epoch_ssims) else 'N/A'
                history_writer.writerow([epoch_num + 1, train_loss, val_psnr_epoch, val_ssim_epoch])
        print(f"Training history for trial {trial_id} saved to {history_csv_path}")
    except Exception as e:
        print(f"Error saving training history for trial {trial_id} to {history_csv_path}: {e}")

    # --- Generate and Save a Random Validation Sample Image ---
    if val_dataset and len(val_dataset) > 0:
        current_model.eval() # Set model to evaluation mode
        
        random_val_idx = random.randint(0, len(val_dataset) - 1)
        fbp_input_val, target_val, ground_truth_val = val_dataset[random_val_idx]
        
        # Prepare input for model: add batch dimension and move to device
        # Input from dataset is [C, H, W], model expects [N, C, H, W]
        fbp_input_model = fbp_input_val.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            reconstructed_output_val = current_model(fbp_input_model)
        
        # Move to CPU, remove batch & channel dims, convert to NumPy for saving
        fbp_image_np = fbp_input_val.squeeze().cpu().numpy()
        reconstructed_image_np = reconstructed_output_val.squeeze().cpu().numpy()
        ground_truth_image_np = ground_truth_val.squeeze().cpu().numpy()
        
        # Define save paths
        fbp_save_path = os.path.join(save_directory, f"{model_save_name_stem}_val_sample_input.png")
        output_save_path = os.path.join(save_directory, f"{model_save_name_stem}_val_sample_reconstructed.png")
        gt_save_path = os.path.join(save_directory, f"{model_save_name_stem}_val_sample_ground_truth.png")
        
        # Save images
        plt.imsave(fbp_save_path, fbp_image_np, cmap='gray')
        plt.imsave(output_save_path, reconstructed_image_np, cmap='gray')
        plt.imsave(gt_save_path, ground_truth_image_np, cmap='gray')
        print(f"Saved validation sample images for trial {trial_id} to {save_directory}")

        current_model.train() # Set model back to training mode (good practice)

    print(f"Finished HPO Trial {trial_id}. Final Val PSNR: {final_val_psnr:.4f}, Final Val SSIM: {final_val_ssim:.4f}")
    return final_val_psnr, final_val_ssim


if __name__ == '__main__':
    # Ensure the base directory for all trained models exists
    os.makedirs(TRAINED_MODELS_BASE_DIR, exist_ok=True) 
    
    hpo_results = []

    print(f"--- Starting Hyperparameter Optimization for {FIXED_MODEL_TYPE_HPO} with {FIXED_N_VIEWS_HPO} views ---")
    print(f"--- Running {N_HPO_TRIALS} random search trials using {NUM_EPOCHS} epochs each ---")

    for trial_num in range(1, N_HPO_TRIALS + 1):
        # Sample hyperparameters for DnCNN
        lr_sample = 10**np.random.uniform(-5, -3)  # Log-uniform: 1e-5 to 1e-3
        batch_size_sample = random.choice([8, 12]) 
        num_layers_sample = random.randint(5, 17) 
        num_features_sample = random.choice([32, 48, 64, 80, 96])

        val_psnr, val_ssim = run_single_hpo_trial(
            trial_id=trial_num,
            model_type_arg=FIXED_MODEL_TYPE_HPO,
            n_views_arg=FIXED_N_VIEWS_HPO,
            lr_hpo=lr_sample,
            batch_size_hpo=batch_size_sample,
            num_layers_dncnn_hpo=num_layers_sample,
            num_features_dncnn_hpo=num_features_sample
        )

        hpo_results.append({
            "trial": trial_num,
            "lr": lr_sample,
            "batch_size": batch_size_sample,
            "num_layers": num_layers_sample,
            "num_features": num_features_sample,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim
        })

        if hpo_results:
            best_ssim_run = max(hpo_results, key=lambda x: x["val_ssim"])
            best_psnr_run = max(hpo_results, key=lambda x: x["val_psnr"])
            print(f"\nBest Val SSIM after trial {trial_num}: {best_ssim_run['val_ssim']:.4f} (Config: Trial {best_ssim_run['trial']})")
            print(f"Best Val PSNR after trial {trial_num}: {best_psnr_run['val_psnr']:.4f} (Config: Trial {best_psnr_run['trial']})")
        print(f"--- Completed HPO Trial {trial_num}/{N_HPO_TRIALS} ---\n")


    hpo_summary_parent_dir = os.path.join(TRAINED_MODELS_BASE_DIR, f"views_{FIXED_N_VIEWS_HPO}", FIXED_MODEL_TYPE_HPO)
    os.makedirs(hpo_summary_parent_dir, exist_ok=True) 
    hpo_csv_filename = f"hpo_ALL_TRIALS_summary_{FIXED_MODEL_TYPE_HPO}_views{FIXED_N_VIEWS_HPO}.csv"
    hpo_csv_path = os.path.join(hpo_summary_parent_dir, hpo_csv_filename)

    try:
        sorted_hpo_results = sorted(hpo_results, key=lambda x: x["val_ssim"], reverse=True)
        with open(hpo_csv_path, 'w', newline='') as csvfile:
            fieldnames = ["trial", "lr", "batch_size", "num_layers", "num_features", "val_psnr", "val_ssim"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for res in sorted_hpo_results:
                writer.writerow(res)
        print(f"\nFull HPO results summary saved to {hpo_csv_path}")
    except Exception as e:
        print(f"Error saving HPO results summary to CSV: {e}")

    print("\n--- Final HPO Results (Sorted by Validation SSIM) ---")
    for res in sorted_hpo_results: 
        print(f"Trial {res['trial']}: LR={res['lr']:.6f}, BS={res['batch_size']}, Layers={res['num_layers']}, Feats={res['num_features']} -> Val PSNR: {res['val_psnr']:.4f}, Val SSIM: {res['val_ssim']:.4f}")
    
    print("\n--- Final HPO Results (Sorted by Validation PSNR) ---")
    for res in sorted(hpo_results, key=lambda x: x["val_psnr"], reverse=True):
        print(f"Trial {res['trial']}: LR={res['lr']:.6f}, BS={res['batch_size']}, Layers={res['num_layers']}, Feats={res['num_features']} -> Val PSNR: {res['val_psnr']:.4f}, Val SSIM: {res['val_ssim']:.4f}")

    print("\n--- HPO Process Completed ---")