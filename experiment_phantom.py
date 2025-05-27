import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import csv # Import the csv module

from utils.models import UNet, DnCNN, REDNet 
from utils.prepare_data import create_phantom_dataset, ReconstructionDataset
from utils.train import train_model, DEVICE 

N_VIEWS_LIST = [1024, 512, 256, 128]
TOTAL_PHANTOMS_PER_VIEW_SETTING = 1000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

NOISE_LEVEL = 0.05 
K_SPLITS_NOISE2INVERSE = 4

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 
BATCH_SIZE = 16 

def run_experiment():
    print(f"Starting experiment on DEVICE: {DEVICE}")

    for n_views in N_VIEWS_LIST:
        print(f"\n--- Processing for {n_views} views ---")

        print(f"Generating {TOTAL_PHANTOMS_PER_VIEW_SETTING} phantoms for {n_views} views...")
        full_dataset_list = create_phantom_dataset(
            n=TOTAL_PHANTOMS_PER_VIEW_SETTING,
            n_views=n_views,
            noise_level=NOISE_LEVEL,
            k_splits=K_SPLITS_NOISE2INVERSE
        )

        if not full_dataset_list:
            print(f"Failed to generate dataset for {n_views} views. Skipping.")
            continue
        
        num_train = int(TRAIN_RATIO * len(full_dataset_list))
        num_val = int(VAL_RATIO * len(full_dataset_list))
        
        indices = np.random.permutation(len(full_dataset_list))
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train : num_train + num_val]
        test_indices = indices[num_train + num_val :] 

        train_data_list = [full_dataset_list[i] for i in train_indices]
        val_data_list = [full_dataset_list[i] for i in val_indices]
        
        train_dataset = ReconstructionDataset(train_data_list)
        val_dataset = ReconstructionDataset(val_data_list)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model_architectures = {
            "UNet": UNet(in_channels=1, out_channels=1),
            "DnCNN": DnCNN(in_channels=1, out_channels=1),
            "REDNet": REDNet(in_channels=1, out_channels=1)
        }

        for model_name, model_instance_generator in model_architectures.items():
            print(f"\nTraining {model_name} for {n_views} views...")
            
            current_model = model_instance_generator 
            current_model.to(DEVICE)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(current_model.parameters(), lr=LEARNING_RATE)
            
            save_directory = f"trained_models/views_{n_views}"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            
            epoch_losses, epoch_psnrs, epoch_ssims = train_model(
                model=current_model,
                dataloader=train_loader, 
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=NUM_EPOCHS,
                model_name=f"{model_name}_views_{n_views}", 
                save_path=save_directory
            )
            print(f"Finished training {model_name} for {n_views} views.")

            # Save training history to CSV
            history_csv_filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_views_{n_views}_training_history.csv"
            history_csv_path = os.path.join(save_directory, history_csv_filename)
            
            with open(history_csv_path, 'w', newline='') as csvfile:
                history_writer = csv.writer(csvfile)
                history_writer.writerow(['Epoch', 'Loss', 'PSNR', 'SSIM'])
                for epoch_num in range(NUM_EPOCHS):
                    history_writer.writerow([
                        epoch_num + 1, 
                        epoch_losses[epoch_num], 
                        epoch_psnrs[epoch_num], 
                        epoch_ssims[epoch_num]
                    ])
            print(f"Training history saved to {history_csv_path}")

    print("\n--- Experiment Finished ---")

if __name__ == '__main__':
    run_experiment()