import torch
import torch.nn as nn
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from config import DEVICE, training_losses_history

def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name="Model", save_path="."):
    model.to(DEVICE)
    model.train()
    epoch_losses = []
    print(f"\n--- Training {model_name} ---")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % (max(1, len(dataloader) // 5)) == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}")

    model_save_name = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_trained.pth"
    model_save_path = os.path.join(save_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Finished Training {model_name}. Model saved to {model_save_path}")

    training_losses_history[model_name] = epoch_losses
    return epoch_losses

def evaluate_denoising_metrics(denoised_img, ground_truth_img, noisy_img):
    def normalize_for_metrics(img):
        if img is None: return None
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min) if img_max > img_min else img

    gt_norm = normalize_for_metrics(ground_truth_img)
    denoised_norm = normalize_for_metrics(denoised_img)
    noisy_norm = normalize_for_metrics(noisy_img)

    if gt_norm is None or noisy_norm is None:
        print("Cannot calculate noisy metrics: ground truth or noisy image is None.")
        psnr_noisy, ssim_noisy = np.nan, np.nan
    else:
        psnr_noisy = psnr(gt_norm, noisy_norm, data_range=1.0)
        ssim_noisy = ssim(gt_norm, noisy_norm, data_range=1.0, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    print(f"Noisy Image: PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")

    if gt_norm is None or denoised_norm is None:
        print("Cannot calculate denoised metrics: ground truth or denoised image is None.")
        psnr_denoised, ssim_denoised = np.nan, np.nan
    else:
        psnr_denoised = psnr(gt_norm, denoised_norm, data_range=1.0)
        ssim_denoised = ssim(gt_norm, denoised_norm, data_range=1.0, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    print(f"Denoised Image: PSNR={psnr_denoised:.2f} dB, SSIM={ssim_denoised:.4f}")

    return psnr_denoised, ssim_denoised, psnr_noisy, ssim_noisy