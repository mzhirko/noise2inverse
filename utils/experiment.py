import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import csv
from utils.models import UNet, DnCNN, REDNet
from utils.prepare_data import ReconstructionDataset
from utils.train import train_model, DEVICE

from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric
import torchvision.utils as vutils

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 100
DEFAULT_TRAINED_MODELS_BASE_DIR = "./trained_models"
DEFAULT_LEARNING_RATE = 5e-4

def evaluate_model(
    model,
    test_loader,
    evaluation_dir,
    device,
    model_type_arg,
    n_views_arg
):
    print(f"\nStarting evaluation for Model: {model_type_arg}, Views: {n_views_arg} on DEVICE: {device}")
    model.eval()

    os.makedirs(evaluation_dir, exist_ok=True)
    comparison_images_dir = os.path.join(evaluation_dir, "comparison_images")
    os.makedirs(comparison_images_dir, exist_ok=True)

    all_psnrs = []
    all_ssims = []
    
    with torch.no_grad():
        for i, (batch_input_imgs, _, batch_original_imgs) in enumerate(test_loader):
            batch_input_imgs = batch_input_imgs.to(device)
            batch_original_imgs = batch_original_imgs.to(device)

            outputs = model(batch_input_imgs)

            outputs_np = outputs.cpu().numpy()
            original_imgs_np = batch_original_imgs.cpu().numpy()
            
            for k in range(outputs_np.shape[0]):
                output_img_k = np.squeeze(outputs_np[k])
                original_img_k = np.squeeze(original_imgs_np[k])

                data_r = 1.0
                
                current_psnr = psnr_metric(original_img_k, output_img_k, data_range=data_r)
                
                min_dim_k = min(original_img_k.shape[-2:])
                win_size_k = min(7, min_dim_k if min_dim_k % 2 != 0 else min_dim_k -1)

                if win_size_k < 2 :
                    current_ssim = 0.0 
                else:
                    current_ssim = ssim_metric(original_img_k, output_img_k, data_range=data_r, win_size=win_size_k, channel_axis=None)
                
                all_psnrs.append(current_psnr)
                all_ssims.append(current_ssim)

                input_sample_to_save = batch_input_imgs[k].cpu()
                original_sample_to_save = batch_original_imgs[k].cpu()
                output_sample_to_save = outputs[k].cpu()

                image_grid = vutils.make_grid(
                    [input_sample_to_save, original_sample_to_save, output_sample_to_save],
                    nrow=3,
                    padding=2,
                    normalize=False 
                )
                
                sample_idx = i * test_loader.batch_size + k
                comparison_filename = f"comparison_sample_{sample_idx}_{model_type_arg}_{n_views_arg}.png"
                comparison_filepath = os.path.join(comparison_images_dir, comparison_filename)
                vutils.save_image(image_grid, comparison_filepath)

    avg_psnr = np.mean(all_psnrs) if all_psnrs else 0
    avg_ssim = np.mean(all_ssims) if all_ssims else 0

    print(f"Evaluation - Average PSNR: {avg_psnr:.4f} (vs Original), Average SSIM: {avg_ssim:.4f} (vs Original)")

    metrics_filename = f"evaluation_metrics_{model_type_arg}_{n_views_arg}.csv"
    metrics_filepath = os.path.join(evaluation_dir, metrics_filename)
    with open(metrics_filepath, 'w', newline='') as csvfile:
        metrics_writer = csv.writer(csvfile)
        metrics_writer.writerow(['Metric', 'Value'])
        metrics_writer.writerow(['Average PSNR (vs Original)', avg_psnr])
        metrics_writer.writerow(['Average SSIM (vs Original)', avg_ssim])
    
    print(f"Evaluation metrics saved to {metrics_filepath}")
    print(f"Comparison images (Input | Original | Output) saved in: {comparison_images_dir}")
    print(f"--- Evaluation for Model: {model_type_arg}, Views: {n_views_arg} Finished ---")

    return avg_psnr, avg_ssim

def run_experiment(
    model_type_arg: str,
    n_views_arg: int,
    full_dataset_list: list,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    trained_models_base_dir: str = DEFAULT_TRAINED_MODELS_BASE_DIR,
    learning_rate: float = DEFAULT_LEARNING_RATE
):
    print(f"Starting experiment for Model: {model_type_arg}, Views: {n_views_arg} on DEVICE: {DEVICE}")

    num_total_samples = len(full_dataset_list)
    num_train = int(train_ratio * num_total_samples)
    num_val = int(val_ratio * num_total_samples)
    
    if num_train == 0:
        print(f"Number of training samples is 0 (train_ratio={train_ratio}). Aborting experiment.")
        return
    if num_val == 0:
        print(f"Number of validation samples is 0 (val_ratio={val_ratio}). Aborting experiment as validation is expected.")
        return
        
    if num_train + num_val > num_total_samples:
        print(f"Sum of train ({num_train}) and validation ({num_val}) samples exceeds total samples ({num_total_samples}). Adjust ratios or dataset size.")
        return

    num_test = num_total_samples - (num_train + num_val)

    indices = np.random.permutation(num_total_samples)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    test_indices = indices[num_train + num_val : num_train + num_val + num_test]
    
    train_data_list = [full_dataset_list[i] for i in train_indices]
    val_data_list = [full_dataset_list[i] for i in val_indices]
    
    train_dataset = ReconstructionDataset(train_data_list)
    val_dataset = ReconstructionDataset(val_data_list)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    optimizer = optim.Adam(current_model.parameters(), lr=learning_rate)
    
    save_directory = os.path.join(trained_models_base_dir, f"views_{n_views_arg}", model_type_arg)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    
    model_save_name_stem = f"{model_type_arg}_views_{n_views_arg}"

    epoch_losses, epoch_psnrs, epoch_ssims = train_model(
        model=current_model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        model_name=model_save_name_stem,
        save_path=save_directory
    )
    print(f"Finished training {model_type_arg} for {n_views_arg} views.")
    print(f"Model saved in directory: {save_directory}")

    history_csv_filename = f"{model_save_name_stem}_training_history.csv"
    history_csv_path = os.path.join(save_directory, history_csv_filename)
    
    with open(history_csv_path, 'w', newline='') as csvfile:
        history_writer = csv.writer(csvfile)
        history_writer.writerow(['Epoch', 'Train_Loss', 'Val_PSNR', 'Val_SSIM'])
        for epoch_num in range(len(epoch_losses)):
            history_writer.writerow([
                epoch_num + 1,
                epoch_losses[epoch_num],
                epoch_psnrs[epoch_num] if epoch_num < len(epoch_psnrs) else 'N/A',
                epoch_ssims[epoch_num] if epoch_num < len(epoch_ssims) else 'N/A'
            ])
    print(f"Training history saved to {history_csv_path}")

    if num_test > 0:
        test_data_list = [full_dataset_list[i] for i in test_indices]
        test_dataset = ReconstructionDataset(test_data_list)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        evaluation_designated_dir = os.path.join(save_directory, "evaluation_results")
        
        evaluate_model(
            model=current_model,
            test_loader=test_loader,
            evaluation_dir=evaluation_designated_dir,
            device=DEVICE,
            model_type_arg=model_type_arg,
            n_views_arg=n_views_arg
        )
    else:
        print("No test data available (num_test <= 0). Skipping evaluation.")

    print(f"\n--- Experiment run for Model: {model_type_arg}, Views: {n_views_arg} Finished ---")