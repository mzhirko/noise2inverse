# main_experiment.py
# Author: Gemini
# Date: May 14, 2025
# Description:
# This script provides a foundational framework for the project "Impact of Network Architecture on Noise2Inverse".
# It includes:
# 1. Simplified 2D phantom generation.
# 2. CT simulation (projection, noise, FBP reconstruction) using scikit-image.
# 3. Basic PyTorch implementations of U-Net and DnCNN (2D).
# 4. Conceptual structure for Noise2Inverse data loading and training.
# 5. Evaluation metrics (PSNR, SSIM).
#
# This code is a starting point and requires significant expansion and adaptation
# for the full research project, especially for 3D phantom generation,
# sophisticated data handling, and the complete Noise2Inverse training pipeline.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.draw import disk
import matplotlib.pyplot as plt
import random

# --- Configuration ---
PHANTOM_SIZE = 128  # Size of the phantom and reconstructed images
N_PROJECTIONS = 360 # Number of projection angles
NOISE_TYPE = 'poisson' # 'gaussian' or 'poisson'
NOISE_LEVEL_GAUSSIAN = 0.1 # Std dev for Gaussian noise
NOISE_LEVEL_POISSON_LAM = 30 # Lambda for Poisson noise (related to photon count)
MODEL_INPUT_CHANNELS = 1
MODEL_OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 # As per plan, keep fixed for comparison (adjust as needed)
BATCH_SIZE = 4
K_SPLITS_NOISE2INVERSE = 4 # Number of splits for Noise2Inverse data (e.g., 2 or 4)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Phantom Generation ---
def create_simple_phantom(size=128):
    """
    Creates a simple 2D phantom with a few geometric shapes.
    This needs to be replaced with a 3D human-like phantom generator
    as per the experimental plan.
    """
    image = np.zeros((size, size), dtype=np.float32)

    # Large circle (background tissue)
    rr, cc = disk((size // 2, size // 2), radius=size // 2.2, shape=image.shape)
    image[rr, cc] = 0.5

    # Smaller, denser circles (internal structures)
    for _ in range(3):
        r_pos = random.randint(int(size*0.2), int(size*0.8))
        c_pos = random.randint(int(size*0.2), int(size*0.8))
        rad = random.randint(int(size*0.05), int(size*0.15))
        val = random.uniform(0.7, 1.0)
        rr, cc = disk((r_pos, c_pos), radius=rad, shape=image.shape)
        image[rr, cc] = val
    
    # Add some finer details (e.g., small high-contrast points)
    for _ in range(5):
        r_pos = random.randint(int(size*0.1), int(size*0.9))
        c_pos = random.randint(int(size*0.1), int(size*0.9))
        image[r_pos-1:r_pos+1, c_pos-1:c_pos+1] = random.uniform(0.0, 0.3) # some low density spots
        
    return image

# --- 2. CT Simulation ---
def simulate_projections(phantom, n_projections=N_PROJECTIONS):
    """
    Simulates 2D parallel-beam projections (sinogram) from a phantom.
    """
    angles = np.linspace(0., 180., n_projections, endpoint=False)
    sinogram = radon(phantom, theta=angles, circle=True)
    return sinogram, angles

def add_noise(sinogram, noise_type='gaussian', noise_level_gaussian=0.1, noise_level_poisson_lam=30):
    """
    Adds Gaussian or Poisson noise to the sinogram.
    """
    noisy_sinogram = sinogram.copy()
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level_gaussian * np.max(sinogram), sinogram.shape)
        noisy_sinogram += noise
    elif noise_type == 'poisson':
        # Poisson noise is data-dependent. Scale data to simulate photon counts.
        # This is a simplified model. Real Poisson noise depends on I_0 * exp(-projection_value).
        # The Noise2Inverse paper has a more detailed Poisson model.
        # For simplicity here, we scale and apply.
        scaled_sinogram = (noisy_sinogram / np.max(noisy_sinogram)) * noise_level_poisson_lam
        noisy_sinogram = np.random.poisson(np.maximum(0, scaled_sinogram)).astype(np.float32)
        # Rescale back (approximately)
        noisy_sinogram = (noisy_sinogram / noise_level_poisson_lam) * np.max(sinogram)
    else:
        raise ValueError("Unknown noise type. Choose 'gaussian' or 'poisson'.")
    return np.clip(noisy_sinogram, 0, np.max(sinogram)) # Clip to avoid negative values

def reconstruct_fbp(sinogram, angles):
    """
    Reconstructs an image from a sinogram using Filtered Backprojection (FBP).
    """
    # Ram-Lak filter is default in iradon
    reconstruction_fbp = iradon(sinogram, theta=angles, circle=True)
    return reconstruction_fbp

# --- 3. CNN Architectures (PyTorch) ---

# U-Net Implementation (Simplified 2D)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1): # Conv-BatchNorm-ReLU
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512)) # 512 (from upconv) + 512 (from enc4 skip)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256)) # 256 (from upconv) + 256 (from enc3 skip)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128)) # 128 (from upconv) + 128 (from enc2 skip)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))   # 64 (from upconv) + 64 (from enc1 skip)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1) # Skip connection
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1) # Skip connection
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1) # Skip connection
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1) # Skip connection
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out

# DnCNN Implementation (Simplified 2D)
class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers (Conv + BN + ReLU)
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        # Last layer
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # DnCNN learns the residual (noise)
        residual = self.dncnn(x)
        return x - residual # Subtract learned noise from input to get denoised image


# --- 4. Noise2Inverse Data Handling and Training (Conceptual) ---
class Noise2InverseDataset(Dataset):
    def __init__(self, list_of_noisy_sinograms, angles, k_splits=K_SPLITS_NOISE2INVERSE, strategy='X:1'):
        """
        Dataset for Noise2Inverse.
        Args:
            list_of_noisy_sinograms (list): List of full noisy sinograms (each is a NumPy array).
            angles (np.array): Projection angles corresponding to the sinograms.
            k_splits (int): Number of ways to split each sinogram (e.g., 2 or 4).
            strategy (str): 'X:1' means K-1 splits for input, 1 for target.
                             '1:X' means 1 split for input, K-1 for target.
                             (As per Noise2Inverse paper)
        """
        self.list_of_noisy_sinograms = list_of_noisy_sinograms
        self.angles = angles
        self.k_splits = k_splits
        self.strategy = strategy
        self.num_projection_angles = angles.shape[0]

        # Pre-generate all possible input/target reconstruction pairs
        # This can be memory intensive for large datasets.
        # Alternatively, generate on-the-fly in __getitem__.
        self.training_pairs = self._prepare_training_pairs()

    def _prepare_training_pairs(self):
        pairs = []
        for full_noisy_sino in self.list_of_noisy_sinograms:
            # Split the sinogram and angles into K parts
            # Each part contains projections from every Kth angle
            split_sinos = [full_noisy_sino[:, i::self.k_splits] for i in range(self.k_splits)]
            split_angles = [self.angles[i::self.k_splits] for i in range(self.k_splits)]

            # Reconstruct each part
            sub_reconstructions = [reconstruct_fbp(s, a) for s, a in zip(split_sinos, split_angles)]
            
            # Normalize and ensure correct size (e.g., PHANTOM_SIZE x PHANTOM_SIZE)
            sub_reconstructions = [
                resize(recon, (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True, mode='reflect').astype(np.float32)
                for recon in sub_reconstructions
            ]


            # Create input/target pairs based on strategy
            for i in range(self.k_splits): # Iterate through each part as a potential target (or part of target)
                if self.strategy == 'X:1':
                    # Target is one sub-reconstruction
                    target_recon = sub_reconstructions[i]
                    # Input is the mean of the K-1 other sub-reconstructions
                    input_indices = [j for j in range(self.k_splits) if j != i]
                    if not input_indices: continue # Should not happen if k_splits > 1
                    input_recons_to_avg = [sub_reconstructions[j] for j in input_indices]
                    input_recon = np.mean(input_recons_to_avg, axis=0)
                elif self.strategy == '1:X':
                    # Input is one sub-reconstruction
                    input_recon = sub_reconstructions[i]
                    # Target is the mean of the K-1 other sub-reconstructions
                    target_indices = [j for j in range(self.k_splits) if j != i]
                    if not target_indices: continue
                    target_recons_to_avg = [sub_reconstructions[j] for j in target_indices]
                    target_recon = np.mean(target_recons_to_avg, axis=0)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                
                pairs.append((input_recon, target_recon))
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        input_recon, target_recon = self.training_pairs[idx]

        # Convert to PyTorch tensors and add channel dimension
        input_tensor = torch.from_numpy(input_recon).unsqueeze(0)  # (1, H, W)
        target_tensor = torch.from_numpy(target_recon).unsqueeze(0) # (1, H, W)
        return input_tensor, target_tensor

def train_model(model, dataloader, criterion, optimizer, num_epochs, model_name="Model"):
    """
    Basic training loop.
    """
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
            
            if (i + 1) % 10 == 0: # Print every 10 batches
                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")


        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}")

    print(f"Finished Training {model_name}.")
    return epoch_losses

# --- 5. Evaluation Metrics ---
def evaluate_denoising(denoised_img, ground_truth_img, noisy_img):
    """
    Calculates PSNR and SSIM.
    Assumes images are NumPy arrays, normalized to [0, 1] or appropriate range for metrics.
    """
    # Ensure images are in the same range, e.g., [0,1] for skimage.metrics
    def normalize_for_metrics(img):
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img # Avoid division by zero if flat

    gt_norm = normalize_for_metrics(ground_truth_img)
    denoised_norm = normalize_for_metrics(denoised_img)
    noisy_norm = normalize_for_metrics(noisy_img)

    data_range = 1.0 # Since images are normalized to [0,1]

    psnr_denoised = psnr(gt_norm, denoised_norm, data_range=data_range)
    ssim_denoised = ssim(gt_norm, denoised_norm, data_range=data_range, channel_axis=None, win_size=7) # win_size for small images

    psnr_noisy = psnr(gt_norm, noisy_norm, data_range=data_range)
    ssim_noisy = ssim(gt_norm, noisy_norm, data_range=data_range, channel_axis=None, win_size=7)

    print(f"Noisy Image: PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")
    print(f"Denoised Image: PSNR={psnr_denoised:.2f} dB, SSIM={ssim_denoised:.4f}")
    return psnr_denoised, ssim_denoised, psnr_noisy, ssim_noisy


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate Phantom Data
    print("1. Generating phantom data...")
    num_phantoms_for_training = 10 # Small number for demo; increase for real training
    original_phantoms = [create_simple_phantom(PHANTOM_SIZE) for _ in range(num_phantoms_for_training)]
    
    # For evaluation, use one phantom not in training (or a separate test set)
    eval_phantom_gt = create_simple_phantom(PHANTOM_SIZE) # Ground truth for evaluation

    # 2. Simulate CT data for training set
    print("2. Simulating CT data for training...")
    training_noisy_sinograms = []
    training_angles = None
    for i, phantom in enumerate(original_phantoms):
        sinogram, angles = simulate_projections(phantom, N_PROJECTIONS)
        if training_angles is None: training_angles = angles
        noisy_sinogram = add_noise(sinogram, NOISE_TYPE, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
        training_noisy_sinograms.append(noisy_sinogram)
        if i < 2: # Plot first few for sanity check
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title(f"Original Phantom {i+1}")
            axes[0].axis('off')
            axes[1].imshow(sinogram, cmap='gray')
            axes[1].set_title(f"Clean Sinogram {i+1}")
            axes[1].axis('off')
            axes[2].imshow(noisy_sinogram, cmap='gray')
            axes[2].set_title(f"Noisy Sinogram {i+1} ({NOISE_TYPE})")
            axes[2].axis('off')
            plt.tight_layout()
            plt.show()


    # 3. Prepare Noise2Inverse DataLoader
    print("3. Preparing Noise2Inverse DataLoader...")
    # Note: The Noise2InverseDataset reconstructs sub-parts. This can be slow.
    # For a real pipeline, consider pre-calculating and saving these reconstructions,
    # or optimizing the on-the-fly generation.
    n2i_dataset = Noise2InverseDataset(training_noisy_sinograms, training_angles, k_splits=K_SPLITS_NOISE2INVERSE, strategy='X:1')
    if len(n2i_dataset) == 0:
        print("Dataset is empty. Check k_splits and number of phantoms.")
        exit()
        
    n2i_dataloader = DataLoader(n2i_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers > 0 for speed
    print(f"Noise2Inverse dataset size: {len(n2i_dataset)} pairs.")

    # 4. Initialize Models, Loss, Optimizer
    print("4. Initializing models...")
    unet_model = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
    dncnn_model = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)

    criterion = nn.MSELoss() # As per experimental plan (fixed loss)

    # Optimizers (fixed learning rate as per plan)
    optimizer_unet = optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
    optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=LEARNING_RATE)

    # 5. Train Models (Conceptual - this will take time)
    # For a quick test, you might reduce NUM_EPOCHS or dataset size
    # IMPORTANT: Your plan is to compare U-Net vs DnCNN. Train them separately.
    
    # --- Train U-Net ---
    unet_losses = train_model(unet_model, n2i_dataloader, criterion, optimizer_unet, NUM_EPOCHS, "U-Net")

    # --- Train DnCNN ---
    dncnn_losses = train_model(dncnn_model, n2i_dataloader, criterion, optimizer_dncnn, NUM_EPOCHS, "DnCNN")

    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(unet_losses, label="U-Net Training Loss")
    plt.plot(dncnn_losses, label="DnCNN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("Training Convergence Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Evaluation
    print("\n6. Evaluating models...")
    # Simulate noisy data for the evaluation phantom
    eval_sino_gt, eval_angles = simulate_projections(eval_phantom_gt, N_PROJECTIONS)
    eval_sino_noisy = add_noise(eval_sino_gt, NOISE_TYPE, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
    
    # FBP of the full noisy sinogram for evaluation input
    eval_fbp_noisy = reconstruct_fbp(eval_sino_noisy, eval_angles)
    eval_fbp_noisy = resize(eval_fbp_noisy, (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True, mode='reflect').astype(np.float32)
    
    # Ground truth reconstruction (FBP of clean sinogram)
    eval_fbp_gt = reconstruct_fbp(eval_sino_gt, eval_angles)
    eval_fbp_gt = resize(eval_fbp_gt, (PHANTOM_SIZE, PHANTOM_SIZE), anti_aliasing=True, mode='reflect').astype(np.float32)


    # Denoise using trained models
    # For Noise2Inverse, the input to the trained network during inference is typically
    # an average of sub-reconstructions from the *full* noisy measurement,
    # or simply the FBP of the full noisy measurement if the network was trained
    # to map from a less complete reconstruction to a more complete one.
    # Here, for simplicity, we'll use the FBP of the full noisy sinogram as input.
    # The Noise2Inverse paper (Sec III-B, Eq. 20) describes section-wise averaging for output.
    # This part needs careful implementation according to the Noise2Inverse inference strategy.
    # For this demo, we'll apply the model directly to the noisy FBP.

    unet_model.eval()
    dncnn_model.eval()
    with torch.no_grad():
        input_tensor_eval = torch.from_numpy(eval_fbp_noisy).unsqueeze(0).unsqueeze(0).to(DEVICE) # (B, C, H, W)
        
        denoised_unet_tensor = unet_model(input_tensor_eval)
        denoised_unet_img = denoised_unet_tensor.squeeze().cpu().numpy()

        denoised_dncnn_tensor = dncnn_model(input_tensor_eval)
        denoised_dncnn_img = denoised_dncnn_tensor.squeeze().cpu().numpy()

    print("\n--- U-Net Evaluation ---")
    evaluate_denoising(denoised_unet_img, eval_fbp_gt, eval_fbp_noisy)
    
    print("\n--- DnCNN Evaluation ---")
    evaluate_denoising(denoised_dncnn_img, eval_fbp_gt, eval_fbp_noisy)

    # Visual Inspection
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    common_kwargs = {'cmap': 'gray', 'vmin': np.min(eval_fbp_gt), 'vmax': np.max(eval_fbp_gt)}

    axes[0, 0].imshow(eval_fbp_gt, **common_kwargs)
    axes[0, 0].set_title("Ground Truth FBP (Clean)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(eval_fbp_noisy, **common_kwargs)
    axes[0, 1].set_title(f"Noisy FBP ({NOISE_TYPE})")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.zeros_like(eval_fbp_gt), **common_kwargs) # Placeholder
    axes[0, 2].set_title("Empty")
    axes[0, 2].axis('off')


    axes[1, 0].imshow(denoised_unet_img, **common_kwargs)
    axes[1, 0].set_title("Denoised by U-Net")
    axes[1, 0].axis('off')
    
    # Residual for U-Net (denoised - ground_truth)
    residual_unet = denoised_unet_img - eval_fbp_gt
    axes[1, 1].imshow(residual_unet, cmap='coolwarm', vmin=-0.3*np.max(eval_fbp_gt), vmax=0.3*np.max(eval_fbp_gt))
    axes[1, 1].set_title("Residual (U-Net - GT)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_dncnn_img, **common_kwargs)
    axes[1, 2].set_title("Denoised by DnCNN")
    axes[1, 2].axis('off')
    
    # Add another row for DnCNN residual if needed or combine plots
    plt.tight_layout()
    plt.suptitle("Visual Comparison of Denoising Results", fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

    # Residual for DnCNN (denoised - ground_truth)
    residual_dncnn = denoised_dncnn_img - eval_fbp_gt
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(residual_unet, cmap='coolwarm', vmin=-0.3*np.max(eval_fbp_gt), vmax=0.3*np.max(eval_fbp_gt))
    plt.title("Residual (U-Net - GT)")
    plt.axis('off')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(residual_dncnn, cmap='coolwarm', vmin=-0.3*np.max(eval_fbp_gt), vmax=0.3*np.max(eval_fbp_gt))
    plt.title("Residual (DnCNN - GT)")
    plt.axis('off')
    plt.colorbar()
    plt.tight_layout()
    plt.show()


    print("\nScript finished. Remember to expand and refine for your research.")

