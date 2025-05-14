# main_experiment.py
# Author: Gemini
# Date: May 14, 2025
# Description:
# This script provides a foundational framework for the project "Impact of Network Architecture on Noise2Inverse".
# Updates:
# - Integrated 3D phantom generation logic (takes a 2D slice for current 2D pipeline).
# - Added fine-scale details to the 3D phantom.
# - Implemented argparse for modular execution.
#
# It includes:
# 1. 3D human-like phantom generation (outputs a 2D slice for simulation).
# 2. CT simulation (projection, noise, FBP reconstruction) using scikit-image.
# 3. Basic PyTorch implementations of U-Net and DnCNN (2D).
# 4. Conceptual structure for Noise2Inverse data loading and training.
# 5. Evaluation metrics (PSNR, SSIM).
#
# This code is a starting point and requires significant expansion and adaptation
# for the full research project, especially for full 3D CT simulation and 3D models.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.draw import disk # Replaced by 3D drawing
import matplotlib.pyplot as plt
import random
import argparse # Added for command-line arguments
import os # For saving/loading data potentially

# --- Configuration ---
PHANTOM_SIZE_XY = 128  # Size of the 2D slice taken from 3D phantom
PHANTOM_DEPTH_Z = 64   # Depth of the 3D phantom
N_PROJECTIONS = 180 # Number of projection angles (reduced for faster demo)
NOISE_TYPE = 'poisson' # 'gaussian' or 'poisson'
NOISE_LEVEL_GAUSSIAN = 0.1 # Std dev for Gaussian noise
NOISE_LEVEL_POISSON_LAM = 30 # Lambda for Poisson noise (related to photon count)
MODEL_INPUT_CHANNELS = 1
MODEL_OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 # Reduced for faster demo; As per plan, keep fixed for comparison (adjust as needed)
BATCH_SIZE = 2 # Reduced for faster demo
K_SPLITS_NOISE2INVERSE = 2 # Number of splits for Noise2Inverse data (e.g., 2 or 4)

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

# --- 1. 3D Phantom Generation ---

def create_voxel_sphere(center_x, center_y, center_z, radius, array_shape, value=1.0):
    """Creates a sphere in a 3D numpy array."""
    x, y, z = np.ogrid[:array_shape[0], :array_shape[1], :array_shape[2]]
    mask = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2 <= radius**2
    return mask, value

def create_voxel_ellipsoid(center_x, center_y, center_z, rx, ry, rz, array_shape, value=1.0):
    """Creates an ellipsoid in a 3D numpy array."""
    x, y, z = np.ogrid[:array_shape[0], :array_shape[1], :array_shape[2]]
    mask = ((x - center_x)/rx)**2 + ((y - center_y)/ry)**2 + ((z - center_z)/rz)**2 <= 1
    return mask, value

def create_voxel_cuboid(center_x, center_y, center_z, sx, sy, sz, array_shape, value=1.0):
    """Creates a cuboid in a 3D numpy array."""
    x0, x1 = center_x - sx//2, center_x + sx//2
    y0, y1 = center_y - sy//2, center_y + sy//2
    z0, z1 = center_z - sz//2, center_z + sz//2
    mask = np.zeros(array_shape, dtype=bool)
    mask[int(x0):int(x1), int(y0):int(y1), int(z0):int(z1)] = True
    return mask, value

def add_fine_scale_details(phantom_array, num_features=10, max_size=3, value_range=(1.5, 2.0)):
    """Adds small, high-contrast features (spheres) within existing structures."""
    shape = phantom_array.shape
    # Add features only where phantom already has some structure (e.g., > 0.1)
    existing_structure_indices = np.argwhere(phantom_array > 0.1)
    if len(existing_structure_indices) == 0:
        print("Warning: No existing structures found to add fine details to.")
        return phantom_array

    for _ in range(num_features):
        # Pick a random location within an existing structure
        rand_idx = random.choice(existing_structure_indices)
        center_x, center_y, center_z = rand_idx[0], rand_idx[1], rand_idx[2]
        
        radius = random.uniform(1, max_size)
        value = random.uniform(value_range[0], value_range[1])
        
        # Ensure feature is within bounds
        if (center_x - radius < 0 or center_x + radius >= shape[0] or
            center_y - radius < 0 or center_y + radius >= shape[1] or
            center_z - radius < 0 or center_z + radius >= shape[2]):
            continue

        mask, val = create_voxel_sphere(center_x, center_y, center_z, radius, shape, value)
        phantom_array[mask] = val # Add or overwrite with higher value
    return phantom_array


def generate_3d_human_phantom_array(xy_size=PHANTOM_SIZE_XY, z_depth=PHANTOM_DEPTH_Z):
    """
    Generates a 3D human-like phantom as a NumPy array.
    Adapted from user-provided phantom.py, scaled to voxel space.
    """
    phantom = np.zeros((xy_size, xy_size, z_depth), dtype=np.float32)
    
    # Simplified proportions for voxel space
    head_radius = xy_size // 8
    head_center_x, head_center_y = xy_size // 2, xy_size // 2
    head_center_z = z_depth - head_radius - z_depth // 20 # Place head towards the top

    mask, val = create_voxel_sphere(head_center_x, head_center_y, head_center_z, head_radius, phantom.shape, value=0.7)
    phantom[mask] = val

    # Torso (ellipsoid)
    torso_rx, torso_ry, torso_rz = xy_size // 5, xy_size // 7, z_depth // 3
    torso_center_x, torso_center_y = xy_size // 2, xy_size // 2
    torso_center_z = head_center_z - head_radius - torso_rz 
    mask, val = create_voxel_ellipsoid(torso_center_x, torso_center_y, torso_center_z,
                                       torso_rx, torso_ry, torso_rz, phantom.shape, value=0.5)
    phantom[mask] = val
    
    # Limbs (cuboids) - very simplified
    limb_width, limb_thickness = xy_size // 15, xy_size // 15
    arm_length = z_depth // 3
    leg_length = z_depth // 2.5

    # Left Arm
    mask, val = create_voxel_cuboid(xy_size // 2 - torso_rx - limb_width //2 , xy_size // 2, torso_center_z,
                                    limb_width, limb_thickness, arm_length, phantom.shape, value=0.4)
    phantom[mask] = val
    # Right Arm
    mask, val = create_voxel_cuboid(xy_size // 2 + torso_rx + limb_width //2, xy_size // 2, torso_center_z,
                                    limb_width, limb_thickness, arm_length, phantom.shape, value=0.4)
    phantom[mask] = val

    # Add some internal structures (spheres of different densities)
    internal_structures = [
        (head_center_x, head_center_y, head_center_z, head_radius // 2, 0.9), # Brain-like
        (torso_center_x + torso_rx//3, torso_center_y, torso_center_z, torso_ry // 3, 0.8), # Organ 1
        (torso_center_x - torso_rx//3, torso_center_y, torso_center_z, torso_ry // 3, 0.85),# Organ 2
    ]
    for cx, cy, cz, rad, v_val in internal_structures:
        # Ensure internal structures are within bounds and main shape
        # This check is simplified; more robust checks would ensure they are fully contained.
        if 0 < cx < xy_size and 0 < cy < xy_size and 0 < cz < z_depth:
             mask, _ = create_voxel_sphere(cx, cy, cz, rad, phantom.shape, v_val)
             phantom[mask] = v_val # Assign value

    # Add fine-scale details
    phantom = add_fine_scale_details(phantom, num_features=20, max_size=xy_size//32, value_range=(1.0, 1.5))

    return phantom

def get_2d_slice_from_3d_phantom(phantom_3d, slice_index=None):
    """Extracts a 2D slice (e.g., central) from the 3D phantom."""
    if slice_index is None:
        slice_index = phantom_3d.shape[2] // 2 # Central slice along Z-axis
    
    phantom_2d = phantom_3d[:, :, slice_index].copy() # Take a copy
    # Ensure the 2D slice has the target XY dimensions if different
    if phantom_2d.shape[0] != PHANTOM_SIZE_XY or phantom_2d.shape[1] != PHANTOM_SIZE_XY:
         phantom_2d = resize(phantom_2d, (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY), anti_aliasing=True, mode='reflect')
    return phantom_2d.astype(np.float32)


# --- 2. CT Simulation ---
def simulate_projections(phantom_2d, n_projections=N_PROJECTIONS):
    """
    Simulates 2D parallel-beam projections (sinogram) from a 2D phantom slice.
    """
    angles = np.linspace(0., 180., n_projections, endpoint=False)
    sinogram = radon(phantom_2d, theta=angles, circle=True)
    return sinogram, angles

def add_noise(sinogram, noise_type='gaussian', noise_level_gaussian=0.1, noise_level_poisson_lam=30):
    """
    Adds Gaussian or Poisson noise to the sinogram.
    """
    noisy_sinogram = sinogram.copy()
    max_sino_val = np.max(sinogram)
    if max_sino_val == 0: max_sino_val = 1.0 # Avoid division by zero for empty sinograms

    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level_gaussian * max_sino_val, sinogram.shape)
        noisy_sinogram += noise
    elif noise_type == 'poisson':
        scaled_sinogram = (noisy_sinogram / max_sino_val) * noise_level_poisson_lam
        noisy_sinogram = np.random.poisson(np.maximum(0, scaled_sinogram)).astype(np.float32)
        noisy_sinogram = (noisy_sinogram / noise_level_poisson_lam) * max_sino_val
    else:
        raise ValueError("Unknown noise type. Choose 'gaussian' or 'poisson'.")
    return np.clip(noisy_sinogram, 0, None) # Clip negative values, no upper bound needed for Poisson typically

def reconstruct_fbp(sinogram, angles):
    """
    Reconstructs an image from a sinogram using Filtered Backprojection (FBP).
    """
    reconstruction_fbp = iradon(sinogram, theta=angles, circle=True)
    # Ensure output size matches PHANTOM_SIZE_XY
    if reconstruction_fbp.shape[0] != PHANTOM_SIZE_XY or reconstruction_fbp.shape[1] != PHANTOM_SIZE_XY:
        reconstruction_fbp = resize(reconstruction_fbp, (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY), 
                                    anti_aliasing=True, mode='reflect')
    return reconstruction_fbp.astype(np.float32)

# --- 3. CNN Architectures (PyTorch) --- unchanged from previous version ---
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False), # bias=False with BN
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        d4 = self.upconv4(b); d4 = torch.cat((e4, d4), dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat((e3, d3), dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat((e2, d2), dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat((e1, d1), dim=1); d1 = self.dec1(d1)
        return self.out_conv(d1)

class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        residual = self.dncnn(x)
        return x - residual
# --- End of unchanged CNN Architectures ---

# --- 4. Noise2Inverse Data Handling and Training (Conceptual) ---
class Noise2InverseDataset(Dataset):
    def __init__(self, list_of_noisy_sinograms, angles, k_splits=K_SPLITS_NOISE2INVERSE, strategy='X:1', recon_size=PHANTOM_SIZE_XY):
        self.list_of_noisy_sinograms = list_of_noisy_sinograms
        self.angles = angles
        self.k_splits = k_splits
        self.strategy = strategy
        self.recon_size = recon_size
        self.training_pairs = self._prepare_training_pairs()

    def _prepare_training_pairs(self):
        pairs = []
        for full_noisy_sino in self.list_of_noisy_sinograms:
            split_sinos = [full_noisy_sino[:, i::self.k_splits] for i in range(self.k_splits)]
            split_angles = [self.angles[i::self.k_splits] for i in range(self.k_splits)]
            
            sub_reconstructions_raw = [reconstruct_fbp(s, a) for s, a in zip(split_sinos, split_angles)]
            sub_reconstructions = [
                resize(recon, (self.recon_size, self.recon_size), anti_aliasing=True, mode='reflect').astype(np.float32)
                for recon in sub_reconstructions_raw
            ]

            for i in range(self.k_splits):
                if self.strategy == 'X:1':
                    target_recon = sub_reconstructions[i]
                    input_indices = [j for j in range(self.k_splits) if j != i]
                    if not input_indices: continue
                    input_recon = np.mean([sub_reconstructions[j] for j in input_indices], axis=0)
                elif self.strategy == '1:X':
                    input_recon = sub_reconstructions[i]
                    target_indices = [j for j in range(self.k_splits) if j != i]
                    if not target_indices: continue
                    target_recon = np.mean([sub_reconstructions[j] for j in target_indices], axis=0)
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
                pairs.append((input_recon, target_recon))
        return pairs

    def __len__(self):
        return len(self.training_pairs)

    def __getitem__(self, idx):
        input_recon, target_recon = self.training_pairs[idx]
        input_tensor = torch.from_numpy(input_recon).unsqueeze(0)
        target_tensor = torch.from_numpy(target_recon).unsqueeze(0)
        return input_tensor, target_tensor

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
            if (i + 1) % (max(1, len(dataloader) // 5)) == 0: # Print a few times per epoch
                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")
        epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.6f}")
    
    # Save the trained model
    model_save_path = os.path.join(save_path, f"{model_name.lower().replace(' ', '_')}_trained.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Finished Training {model_name}. Model saved to {model_save_path}")
    return epoch_losses

# --- 5. Evaluation Metrics ---
def evaluate_denoising(denoised_img, ground_truth_img, noisy_img):
    def normalize_for_metrics(img):
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min) if img_max > img_min else img
    gt_norm, denoised_norm, noisy_norm = map(normalize_for_metrics, [ground_truth_img, denoised_img, noisy_img])
    data_range = 1.0
    psnr_denoised = psnr(gt_norm, denoised_norm, data_range=data_range)
    ssim_denoised = ssim(gt_norm, denoised_norm, data_range=data_range, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    psnr_noisy = psnr(gt_norm, noisy_norm, data_range=data_range)
    ssim_noisy = ssim(gt_norm, noisy_norm, data_range=data_range, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    print(f"Noisy Image: PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")
    print(f"Denoised Image: PSNR={psnr_denoised:.2f} dB, SSIM={ssim_denoised:.4f}")
    return psnr_denoised, ssim_denoised, psnr_noisy, ssim_noisy

# --- Helper function for argparse ---
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Main Execution with Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise2Inverse Experiment Pipeline")
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate_data', 'train_unet', 'train_dncnn', 'evaluate', 'full_run', 'visualize_phantom'],
                        help='Action to perform.')
    parser.add_argument('--num_train_phantoms', type=int, default=5, help='Number of phantoms for training.')
    parser.add_argument('--num_eval_phantoms', type=int, default=1, help='Number of phantoms for evaluation.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--k_splits', type=int, default=K_SPLITS_NOISE2INVERSE, help='K splits for Noise2Inverse dataset.')
    parser.add_argument('--noise_type_arg', type=str, default=NOISE_TYPE, choices=['gaussian', 'poisson'], help='Type of noise to add.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--plot_data', type=str_to_bool, nargs='?', const=True, default=False, help='Plot generated data samples.')
    parser.add_argument('--model_load_path', type=str, default='.', help='Directory to load trained models from for evaluation.')
    parser.add_argument('--data_save_path', type=str, default='./data', help='Directory to save/load generated data.')

    args = parser.parse_args()

    os.makedirs(args.data_save_path, exist_ok=True) # Ensure data directory exists

    original_phantoms_2d_train = []
    training_noisy_sinograms = []
    training_angles = None
    
    eval_phantoms_gt_2d = []
    eval_sinos_gt = []
    eval_sinos_noisy = []
    eval_fbps_gt = []
    eval_fbps_noisy = []
    eval_angles_list = []


    if args.action in ['generate_data', 'full_run', 'train_unet', 'train_dncnn']:
        print("1. Generating training phantom data...")
        for i in range(args.num_train_phantoms):
            phantom_3d = generate_3d_human_phantom_array(PHANTOM_SIZE_XY, PHANTOM_DEPTH_Z)
            phantom_2d = get_2d_slice_from_3d_phantom(phantom_3d) # Using a 2D slice
            original_phantoms_2d_train.append(phantom_2d)
            
            # Save phantom if generate_data is the sole action
            if args.action == 'generate_data':
                 np.save(os.path.join(args.data_save_path, f"train_phantom_2d_{i}.npy"), phantom_2d)

            if args.plot_data and i < 2 : # Plot first few
                plt.figure(figsize=(6,6))
                plt.imshow(phantom_2d, cmap='gray')
                plt.title(f"Generated 2D Training Phantom Slice {i+1}")
                plt.colorbar()
                plt.show()
        
        print("2. Simulating CT data for training...")
        for i, p_2d in enumerate(original_phantoms_2d_train):
            sinogram, angles_current = simulate_projections(p_2d, N_PROJECTIONS)
            if training_angles is None: training_angles = angles_current
            noisy_sinogram = add_noise(sinogram, args.noise_type_arg, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
            training_noisy_sinograms.append(noisy_sinogram)
            
            if args.action == 'generate_data':
                np.save(os.path.join(args.data_save_path, f"train_sino_noisy_{i}.npy"), noisy_sinogram)
                if i == 0: np.save(os.path.join(args.data_save_path, f"train_angles.npy"), training_angles)


            if args.plot_data and i < 2:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(p_2d, cmap='gray'); axes[0].set_title(f"Orig Phantom Slice {i+1}"); axes[0].axis('off')
                axes[1].imshow(sinogram, cmap='gray'); axes[1].set_title("Clean Sinogram"); axes[1].axis('off')
                axes[2].imshow(noisy_sinogram, cmap='gray'); axes[2].set_title(f"Noisy Sinogram ({args.noise_type_arg})"); axes[2].axis('off')
                plt.show()
        if args.action == 'generate_data':
             print(f"Training data (phantoms, sinograms, angles) saved to {args.data_save_path}")


    if args.action in ['generate_data', 'full_run', 'evaluate']:
        print("1. Generating evaluation phantom data...")
        for i in range(args.num_eval_phantoms):
            phantom_3d_eval = generate_3d_human_phantom_array(PHANTOM_SIZE_XY, PHANTOM_DEPTH_Z)
            phantom_2d_eval_gt = get_2d_slice_from_3d_phantom(phantom_3d_eval)
            eval_phantoms_gt_2d.append(phantom_2d_eval_gt)

            sino_gt, angles_curr_eval = simulate_projections(phantom_2d_eval_gt, N_PROJECTIONS)
            sino_noisy = add_noise(sino_gt, args.noise_type_arg, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
            
            fbp_gt = reconstruct_fbp(sino_gt, angles_curr_eval)
            fbp_noisy = reconstruct_fbp(sino_noisy, angles_curr_eval)

            eval_sinos_gt.append(sino_gt)
            eval_sinos_noisy.append(sino_noisy)
            eval_fbps_gt.append(fbp_gt)
            eval_fbps_noisy.append(fbp_noisy)
            eval_angles_list.append(angles_curr_eval)

            if args.action == 'generate_data':
                np.save(os.path.join(args.data_save_path, f"eval_phantom_2d_gt_{i}.npy"), phantom_2d_eval_gt)
                np.save(os.path.join(args.data_save_path, f"eval_sino_gt_{i}.npy"), sino_gt)
                np.save(os.path.join(args.data_save_path, f"eval_sino_noisy_{i}.npy"), sino_noisy)
                np.save(os.path.join(args.data_save_path, f"eval_fbp_gt_{i}.npy"), fbp_gt)
                np.save(os.path.join(args.data_save_path, f"eval_fbp_noisy_{i}.npy"), fbp_noisy)
                if i == 0: np.save(os.path.join(args.data_save_path, f"eval_angles.npy"), angles_curr_eval)
            
            if args.plot_data and i < 1: # Plot first eval case
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(phantom_2d_eval_gt, cmap='gray'); axes[0].set_title(f"Eval GT Phantom Slice {i+1}"); axes[0].axis('off')
                axes[1].imshow(sino_noisy, cmap='gray'); axes[1].set_title("Eval Noisy Sinogram"); axes[1].axis('off')
                axes[2].imshow(fbp_gt, cmap='gray'); axes[2].set_title("Eval GT FBP"); axes[2].axis('off')
                axes[3].imshow(fbp_noisy, cmap='gray'); axes[3].set_title("Eval Noisy FBP"); axes[3].axis('off')
                plt.show()
        if args.action == 'generate_data':
            print(f"Evaluation data saved to {args.data_save_path}")


    if args.action == 'visualize_phantom':
        print("Visualizing a sample 3D phantom and its 2D slice...")
        phantom_3d_viz = generate_3d_human_phantom_array(PHANTOM_SIZE_XY, PHANTOM_DEPTH_Z)
        phantom_2d_viz = get_2d_slice_from_3d_phantom(phantom_3d_viz)

        fig_3d = plt.figure(figsize=(7,7))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        # Simple 3D scatter plot of non-zero voxels for visualization
        x, y, z = np.where(phantom_3d_viz > 0.1) # Plot points with some density
        c = phantom_3d_viz[x,y,z]
        ax_3d.scatter(x, y, z, c=c, cmap='viridis', s=5)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title('3D Phantom Voxel Visualization (Sample)')
        plt.show()

        plt.figure(figsize=(6,6))
        plt.imshow(phantom_2d_viz, cmap='gray')
        plt.title("2D Slice from 3D Phantom")
        plt.colorbar()
        plt.show()


    if args.action in ['train_unet', 'train_dncnn', 'full_run']:
        # Load data if not generated in this run (assuming 'full_run' generates it)
        if not training_noisy_sinograms: # If list is empty, try loading
            print(f"Loading training data from {args.data_save_path}...")
            try:
                for i in range(args.num_train_phantoms):
                    training_noisy_sinograms.append(np.load(os.path.join(args.data_save_path, f"train_sino_noisy_{i}.npy")))
                training_angles = np.load(os.path.join(args.data_save_path, f"train_angles.npy"))
            except FileNotFoundError:
                print("Error: Training data not found. Please run with --action generate_data first or use full_run.")
                exit()
        
        print("3. Preparing Noise2Inverse DataLoader...")
        n2i_dataset = Noise2InverseDataset(training_noisy_sinograms, training_angles, 
                                           k_splits=args.k_splits, recon_size=PHANTOM_SIZE_XY)
        if len(n2i_dataset) == 0:
            print("Dataset is empty. Check k_splits and number of phantoms.")
            exit()
        n2i_dataloader = DataLoader(n2i_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(f"Noise2Inverse dataset size: {len(n2i_dataset)} pairs.")

        criterion = nn.MSELoss()
        
        if args.action in ['train_unet', 'full_run']:
            print("4a. Initializing U-Net model...")
            unet_model = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_unet = optim.Adam(unet_model.parameters(), lr=args.lr)
            unet_losses = train_model(unet_model, n2i_dataloader, criterion, optimizer_unet, args.epochs, "U-Net", save_path=args.model_load_path)
            if args.plot_data:
                plt.figure(figsize=(10, 5))
                plt.plot(unet_losses, label="U-Net Training Loss")
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("U-Net Training"); plt.legend(); plt.grid(True); plt.show()

        if args.action in ['train_dncnn', 'full_run']:
            print("4b. Initializing DnCNN model...")
            dncnn_model = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=args.lr)
            dncnn_losses = train_model(dncnn_model, n2i_dataloader, criterion, optimizer_dncnn, args.epochs, "DnCNN", save_path=args.model_load_path)
            if args.plot_data:
                plt.figure(figsize=(10, 5))
                plt.plot(dncnn_losses, label="DnCNN Training Loss")
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("DnCNN Training"); plt.legend(); plt.grid(True); plt.show()

    if args.action in ['evaluate', 'full_run']:
        # Load evaluation data if not generated in this run
        if not eval_fbps_gt:
            print(f"Loading evaluation data from {args.data_save_path}...")
            try:
                for i in range(args.num_eval_phantoms):
                    eval_fbps_gt.append(np.load(os.path.join(args.data_save_path, f"eval_fbp_gt_{i}.npy")))
                    eval_fbps_noisy.append(np.load(os.path.join(args.data_save_path, f"eval_fbp_noisy_{i}.npy")))
                # eval_angles = np.load(os.path.join(args.data_save_path, f"eval_angles.npy")) # Not strictly needed for direct FBP eval
            except FileNotFoundError:
                print("Error: Evaluation FBP data not found. Please run with --action generate_data first or use full_run.")
                exit()

        print("\n6. Evaluating models...")
        
        unet_model_eval = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        dncnn_model_eval = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)

        try:
            unet_model_eval.load_state_dict(torch.load(os.path.join(args.model_load_path, "u-net_trained.pth"), map_location=DEVICE))
            print("U-Net model loaded for evaluation.")
        except FileNotFoundError:
            print("U-Net model not found. Skipping U-Net evaluation. Train it first or check path.")
            unet_model_eval = None

        try:
            dncnn_model_eval.load_state_dict(torch.load(os.path.join(args.model_load_path, "dncnn_trained.pth"), map_location=DEVICE))
            print("DnCNN model loaded for evaluation.")
        except FileNotFoundError:
            print("DnCNN model not found. Skipping DnCNN evaluation. Train it first or check path.")
            dncnn_model_eval = None
        
        if unet_model_eval: unet_model_eval.to(DEVICE).eval()
        if dncnn_model_eval: dncnn_model_eval.to(DEVICE).eval()

        for i in range(len(eval_fbps_gt)):
            current_fbp_gt = eval_fbps_gt[i]
            current_fbp_noisy = eval_fbps_noisy[i]
            print(f"\n--- Evaluating on Phantom {i+1} ---")

            denoised_unet_img, denoised_dncnn_img = None, None

            with torch.no_grad():
                input_tensor_eval = torch.from_numpy(current_fbp_noisy).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                if unet_model_eval:
                    denoised_unet_tensor = unet_model_eval(input_tensor_eval)
                    denoised_unet_img = denoised_unet_tensor.squeeze().cpu().numpy()
                    print("\nU-Net Evaluation Results:")
                    evaluate_denoising(denoised_unet_img, current_fbp_gt, current_fbp_noisy)
                
                if dncnn_model_eval:
                    denoised_dncnn_tensor = dncnn_model_eval(input_tensor_eval)
                    denoised_dncnn_img = denoised_dncnn_tensor.squeeze().cpu().numpy()
                    print("\nDnCNN Evaluation Results:")
                    evaluate_denoising(denoised_dncnn_img, current_fbp_gt, current_fbp_noisy)

            # Visual Inspection for the current phantom
            if args.plot_data:
                num_cols = 2 # GT, Noisy
                if denoised_unet_img is not None: num_cols += 1
                if denoised_dncnn_img is not None: num_cols +=1
                
                fig_w = 5 * num_cols
                fig, axes = plt.subplots(1, num_cols, figsize=(fig_w, 5))
                ax_idx = 0
                
                common_kwargs = {'cmap': 'gray', 'vmin': np.min(current_fbp_gt), 'vmax': np.max(current_fbp_gt)}
                axes[ax_idx].imshow(current_fbp_gt, **common_kwargs); axes[ax_idx].set_title("GT FBP"); axes[ax_idx].axis('off'); ax_idx+=1
                axes[ax_idx].imshow(current_fbp_noisy, **common_kwargs); axes[ax_idx].set_title(f"Noisy FBP"); axes[ax_idx].axis('off'); ax_idx+=1
                
                if denoised_unet_img is not None:
                    axes[ax_idx].imshow(denoised_unet_img, **common_kwargs); axes[ax_idx].set_title("U-Net Denoised"); axes[ax_idx].axis('off'); ax_idx+=1
                if denoised_dncnn_img is not None:
                    axes[ax_idx].imshow(denoised_dncnn_img, **common_kwargs); axes[ax_idx].set_title("DnCNN Denoised"); axes[ax_idx].axis('off'); ax_idx+=1
                
                plt.tight_layout()
                plt.suptitle(f"Visual Comparison - Phantom {i+1}", fontsize=16)
                plt.subplots_adjust(top=0.85)
                plt.show()

    print("\nScript finished.")
    if args.action == 'generate_data':
        print(f"Data generation complete. Files saved in {args.data_save_path}")
    print("Remember to expand and refine for your research, especially for full 3D processing.")

