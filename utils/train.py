import torch
import os
import numpy as np
from utils.metrics import calculate_metrics # Assuming this utility is available
import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_name="Model", save_path="./checkpoints"):
    model.to(DEVICE)
    
    train_epoch_losses = []
    val_epoch_psnrs = [] 
    val_epoch_ssims = [] 
    
    calculated_val_epoch_losses = [] 

    print(f"\n--- Training {model_name} ---")
    epochs_pbar = tqdm.tqdm(range(num_epochs), desc=f"Training {model_name} - Epochs")
    for epoch in epochs_pbar:
        model.train()
        running_train_loss = 0.0
        
        train_batches_pbar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for i, (inputs, targets, _) in train_batches_pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            train_batches_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_avg_train_loss = running_train_loss / len(train_dataloader)
        train_epoch_losses.append(epoch_avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        current_epoch_val_psnr_sum = 0.0
        current_epoch_val_ssim_sum = 0.0
        num_val_batches = 0

        val_batches_pbar = tqdm.tqdm(val_dataloader, total=len(val_dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        with torch.no_grad():
            for inputs_val, targets_val, originals_val in val_batches_pbar: 
                inputs_val, targets_val, originals_val = inputs_val.to(DEVICE), targets_val.to(DEVICE), originals_val.to(DEVICE)
                outputs_val = model(inputs_val)

                val_loss = criterion(outputs_val, targets_val) # Loss is calculated against the model's direct target
                running_val_loss += val_loss.item()

                outputs_np = outputs_val.detach().cpu().numpy()
                originals_np = originals_val.detach().cpu().numpy() # Use original phantom/image for metrics
                
                batch_psnr_sum = 0
                batch_ssim_sum = 0
                for j in range(outputs_np.shape[0]): 
                    img_out = np.squeeze(outputs_np[j]) 
                    img_tgt = np.squeeze(originals_np[j]) # Compare output to original phantom/image
                    img_out = np.clip(img_out, 0.0, 1.0)
                    img_tgt = np.clip(img_tgt, 0.0, 1.0) # Ensure original is also clipped if necessary for consistent data_range
                    psnr, ssim = calculate_metrics(img_tgt, img_out, data_range=1.0, win_size=7) 
                    batch_psnr_sum += psnr
                    batch_ssim_sum += ssim
                
                current_epoch_val_psnr_sum += batch_psnr_sum / outputs_np.shape[0] 
                current_epoch_val_ssim_sum += batch_ssim_sum / outputs_np.shape[0]
                num_val_batches += 1
        
        epoch_avg_val_loss = running_val_loss / num_val_batches if num_val_batches > 0 else float('nan')
        epoch_avg_val_psnr = current_epoch_val_psnr_sum / num_val_batches if num_val_batches > 0 else float('nan')
        epoch_avg_val_ssim = current_epoch_val_ssim_sum / num_val_batches if num_val_batches > 0 else float('nan')
        
        calculated_val_epoch_losses.append(epoch_avg_val_loss) 
        val_epoch_psnrs.append(epoch_avg_val_psnr) 
        val_epoch_ssims.append(epoch_avg_val_ssim) 
        
        epochs_pbar.set_postfix(
            train_loss=f"{epoch_avg_train_loss:.4f}", 
            val_loss=f"{epoch_avg_val_loss:.4f}", 
            val_psnr=f"{epoch_avg_val_psnr:.4f}", 
            val_ssim=f"{epoch_avg_val_ssim:.4f}"
        )
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model_save_name = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_trained.pth"
    model_save_path = os.path.join(save_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Finished Training {model_name}. Model saved to {model_save_path}")
    
    return train_epoch_losses, val_epoch_psnrs, val_epoch_ssims