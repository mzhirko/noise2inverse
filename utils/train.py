import torch
import os
import numpy as np
from utils.metrics import calculate_metrics
import tqdm

# Set the default device for training.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_name, save_path):
    """
    Main function for training a model.
    Includes training loop, validation loop, metric calculation, and model saving.
    """
    model.to(DEVICE)
    
    # Lists to store metrics over epochs for later analysis.
    train_epoch_losses = []
    val_epoch_psnrs = [] 
    val_epoch_ssims = [] 
    
    print(f"\n--- Training {model_name} on {DEVICE} ---")
    epochs_pbar = tqdm.tqdm(range(num_epochs), desc=f"Training {model_name}")
    for epoch in epochs_pbar:
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_batches_pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, targets, _ in train_batches_pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_batches_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_epoch_losses.append(running_train_loss / len(train_dataloader))

        # --- Validation Phase ---
        model.eval()
        current_epoch_val_psnr_sum = 0.0
        current_epoch_val_ssim_sum = 0.0
        num_val_batches = len(val_dataloader)

        val_batches_pbar = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for inputs_val, _, originals_val in val_batches_pbar: 
                inputs_val, originals_val = inputs_val.to(DEVICE), originals_val.to(DEVICE)
                outputs_val = model(inputs_val)

                # Calculate metrics (PSNR, SSIM) by comparing the model output to the original, clean image.
                outputs_np = outputs_val.cpu().numpy()
                originals_np = originals_val.cpu().numpy()
                
                batch_psnr_sum = 0
                batch_ssim_sum = 0
                for j in range(outputs_np.shape[0]): 
                    img_out = np.clip(np.squeeze(outputs_np[j]), 0.0, 1.0)
                    img_orig = np.clip(np.squeeze(originals_np[j]), 0.0, 1.0)
                    # Use a fixed window size for consistency in SSIM calculation.
                    psnr, ssim = calculate_metrics(img_orig, img_out, data_range=1.0, win_size=7) 
                    batch_psnr_sum += psnr
                    batch_ssim_sum += ssim
                
                # Accumulate average metrics for the epoch.
                current_epoch_val_psnr_sum += batch_psnr_sum / outputs_np.shape[0] 
                current_epoch_val_ssim_sum += batch_ssim_sum / outputs_np.shape[0]
        
        # Calculate average metrics for the entire validation set.
        epoch_avg_val_psnr = current_epoch_val_psnr_sum / num_val_batches
        epoch_avg_val_ssim = current_epoch_val_ssim_sum / num_val_batches
        val_epoch_psnrs.append(epoch_avg_val_psnr) 
        val_epoch_ssims.append(epoch_avg_val_ssim) 
        
        # Update the main progress bar with the latest metrics.
        epochs_pbar.set_postfix(
            train_loss=f"{train_epoch_losses[-1]:.4f}", 
            val_psnr=f"{epoch_avg_val_psnr:.4f}", 
            val_ssim=f"{epoch_avg_val_ssim:.4f}"
        )
    
    # Save the final trained model state.
    os.makedirs(save_path, exist_ok=True)
    model_save_name = f"{model_name.lower().replace(' ', '_')}_trained.pth"
    model_save_path = os.path.join(save_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nFinished Training {model_name}. Model saved to {model_save_path}")
    
    return train_epoch_losses, val_epoch_psnrs, val_epoch_ssims