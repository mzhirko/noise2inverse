import torch
import os
import numpy as np
from utils.metrics import calculate_metrics 
import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_history = {} 

def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name="Model", save_path="./checkpoints"):
    model.to(DEVICE)
    
    epoch_losses = []
    epoch_psnrs = []
    epoch_ssims = []

    print(f"\n--- Training {model_name} ---")
    epochs_pbar = tqdm(range(num_epochs), desc=f"Training {model_name} - Epochs")
    for epoch in epochs_pbar:
        model.train()
        running_loss = 0.0
        
        train_batches_pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for i, (inputs, targets) in train_batches_pbar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % (max(1, len(dataloader) // 5)) == 0: # Keep existing print for batch loss
                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Training Loss: {loss.item():.6f}")
            train_batches_pbar.set_postfix(loss=loss.item())
        
        epoch_avg_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_avg_loss)

        model.eval()
        current_epoch_psnr_sum = 0.0
        current_epoch_ssim_sum = 0.0
        num_batches_for_metrics = 0

        metrics_batches_pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs} - Eval Metrics", leave=False)
        with torch.no_grad():
            for inputs_eval, targets_eval in metrics_batches_pbar: 
                inputs_eval, targets_eval = inputs_eval.to(DEVICE), targets_eval.to(DEVICE)
                outputs_eval = model(inputs_eval)

                outputs_np = outputs_eval.detach().cpu().numpy()
                targets_np = targets_eval.detach().cpu().numpy()
                
                batch_psnr_sum = 0
                batch_ssim_sum = 0
                for j in range(outputs_np.shape[0]): 
                    img_out = np.squeeze(outputs_np[j]) 
                    img_tgt = np.squeeze(targets_np[j]) 
                    
                    img_out = np.clip(img_out, 0.0, 1.0)
                    img_tgt = np.clip(img_tgt, 0.0, 1.0)

                    psnr, ssim = calculate_metrics(img_tgt, img_out, data_range=1.0, win_size=7) 
                    batch_psnr_sum += psnr
                    batch_ssim_sum += ssim
                
                current_epoch_psnr_sum += batch_psnr_sum / outputs_np.shape[0]
                current_epoch_ssim_sum += batch_ssim_sum / outputs_np.shape[0]
                num_batches_for_metrics += 1
        
        epoch_avg_psnr = current_epoch_psnr_sum / num_batches_for_metrics
        epoch_avg_ssim = current_epoch_ssim_sum / num_batches_for_metrics
        
        epoch_psnrs.append(epoch_avg_psnr)
        epoch_ssims.append(epoch_avg_ssim)
        
        epochs_pbar.set_postfix(loss=epoch_avg_loss, psnr=epoch_avg_psnr, ssim=epoch_avg_ssim)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Training Loss: {epoch_avg_loss:.6f}, Avg Training PSNR: {epoch_avg_psnr:.4f}, Avg Training SSIM: {epoch_avg_ssim:.4f}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    model_save_name = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_trained.pth"
    model_save_path = os.path.join(save_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Finished Training {model_name}. Model saved to {model_save_path}")
    
    training_history[model_name] = {
        'loss': epoch_losses,
        'psnr': epoch_psnrs,
        'ssim': epoch_ssims
    }
    return epoch_losses, epoch_psnrs, epoch_ssims