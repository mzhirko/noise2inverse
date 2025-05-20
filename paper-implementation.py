import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.draw import ellipse, rectangle, polygon, disk
import matplotlib.pyplot as plt
import random
import argparse
import os
#from datetime import datetime # Not used, can be removed
import numpy.fft as np_fft # For MRI simulation

# --- Configuration ---
PHANTOM_SIZE_XY = 128
N_PROJECTIONS = 180 # CT specific
NOISE_TYPE = 'poisson' # CT specific
NOISE_LEVEL_GAUSSIAN = 0.1 # CT specific
NOISE_LEVEL_POISSON_LAM = 30 # CT specific

MODEL_INPUT_CHANNELS = 1
MODEL_OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 # Default, can be overridden by args
BATCH_SIZE = 4 # Default, can be overridden by args
K_SPLITS_NOISE2INVERSE = 2 # CT specific for N2I sinogram splitting

# MRI Specific default parameters
MRI_ACCELERATION_FACTOR = 4.0
MRI_CENTRAL_FRACTION_XY = (0.125, 0.125) # e.g. central 12.5% of kx and ky fully sampled
MRI_KSPACE_NOISE_PERCENT = 5.0 # Noise level as % of max signal in acquired central k-space

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

PLOT_SAVE_DIR = "plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

training_losses_history = {}

# --- MRI Helper Functions ---
def normalize_phantom_for_mri(phantom_data):
    p_min = np.min(phantom_data)
    p_max = np.max(phantom_data)
    if p_max > p_min:
        return (phantom_data - p_min) / (p_max - p_min)
    return np.zeros_like(phantom_data)

def get_kspace_from_image(image_data, use_fftshift=True):
    kspace = np_fft.fft2(image_data)
    if use_fftshift:
        kspace = np_fft.fftshift(kspace)
    return kspace

def get_image_from_kspace(kspace_data, use_ifftshift=True):
    if use_ifftshift:
        kspace_data = np_fft.ifftshift(kspace_data)
    image = np_fft.ifft2(kspace_data)
    return np.abs(image) # Return magnitude

def generate_variable_density_mask(kspace_shape, acceleration_factor, central_fraction_xy=(0.1, 0.1)):
    if not (0 < acceleration_factor):
        raise ValueError("Acceleration factor must be positive.")
    if not (0 <= central_fraction_xy[0] <= 1 and 0 <= central_fraction_xy[1] <= 1):
        raise ValueError("Central fraction must be between 0 and 1.")

    num_rows, num_cols = kspace_shape
    mask = np.zeros(kspace_shape, dtype=bool)

    # Determine central fully sampled region
    center_rows = int(num_rows * central_fraction_xy[0])
    center_cols = int(num_cols * central_fraction_xy[1])
    
    row_start = (num_rows - center_rows) // 2
    row_end = row_start + center_rows
    col_start = (num_cols - center_cols) // 2
    col_end = col_start + center_cols

    mask[row_start:row_end, col_start:col_end] = True
    num_sampled_center = np.sum(mask)

    # Determine how many more points to sample in outer region
    total_points_to_sample = int(num_rows * num_cols / acceleration_factor)
    if total_points_to_sample < num_sampled_center: # Ensure we sample at least the center
        print(f"Warning: Target samples ({total_points_to_sample}) < center samples ({num_sampled_center}). Sampling only center.")
        total_points_to_sample = num_sampled_center
        # Mask is already set for this case
        return mask


    points_to_sample_outer = total_points_to_sample - num_sampled_center
    
    # Create a list of all outer region coordinates
    outer_coords = []
    for r in range(num_rows):
        for c in range(num_cols):
            if not (row_start <= r < row_end and col_start <= c < col_end):
                outer_coords.append((r, c))
    
    if points_to_sample_outer > 0 and len(outer_coords) > 0:
        if points_to_sample_outer > len(outer_coords):
            print(f"Warning: Trying to sample more outer points ({points_to_sample_outer}) than available ({len(outer_coords)}). Sampling all outer points.")
            points_to_sample_outer = len(outer_coords)
            
        random.shuffle(outer_coords)
        chosen_outer_coords = outer_coords[:points_to_sample_outer]
        for r_idx, c_idx in chosen_outer_coords:
            mask[r_idx, c_idx] = True
            
    # print(f"K-space shape: {kspace_shape}, Accel: {acceleration_factor}, Target Samples: {total_points_to_sample}, Actual Samples: {np.sum(mask)}")
    # print(f"Percentage sampled: {100*np.sum(mask)/(num_rows*num_cols):.2f}%")
    return mask

def apply_kspace_mask(kspace_data, mask):
    return kspace_data * mask # Element-wise multiplication

def add_kspace_noise_to_acquired(kspace_masked_data, noise_level_percent_of_max_central_signal=5.0):
    """Adds complex Gaussian noise to non-zero (acquired) k-space samples."""
    noisy_kspace = kspace_masked_data.copy()
    acquired_indices = np.where(kspace_masked_data != 0)

    if acquired_indices[0].size == 0: # No points acquired
        return noisy_kspace

    # Determine max signal in the central acquired region for noise scaling
    # This is a heuristic. A more robust way might use overall signal power.
    center_rows, center_cols = kspace_masked_data.shape[0]//4, kspace_masked_data.shape[1]//4 # Approx central quarter
    row_start, row_end = kspace_masked_data.shape[0]//2 - center_rows//2, kspace_masked_data.shape[0]//2 + center_rows//2
    col_start, col_end = kspace_masked_data.shape[1]//2 - center_cols//2, kspace_masked_data.shape[1]//2 + center_cols//2
    
    central_acquired_signal_mags = np.abs(kspace_masked_data[row_start:row_end, col_start:col_end][kspace_masked_data[row_start:row_end, col_start:col_end] != 0])
    if central_acquired_signal_mags.size > 0:
        max_central_signal = np.max(central_acquired_signal_mags)
    else: # If no central signal acquired, use max of all acquired
        max_central_signal = np.max(np.abs(kspace_masked_data[acquired_indices]))
        if max_central_signal == 0: max_central_signal = 1.0 # Avoid division by zero if all signal is zero


    noise_std_dev = (noise_level_percent_of_max_central_signal / 100.0) * max_central_signal / np.sqrt(2) # per component (real/imag)

    noise_real = np.random.normal(0, noise_std_dev, len(acquired_indices[0]))
    noise_imag = np.random.normal(0, noise_std_dev, len(acquired_indices[0]))
    complex_noise = noise_real + 1j * noise_imag

    noisy_kspace[acquired_indices] += complex_noise
    return noisy_kspace

# --- 1. Dense & Varied 2D Phantom Generation --- (Unchanged from your previous version)
def generate_dense_varied_2d_phantom(size=PHANTOM_SIZE_XY):
    phantom = np.zeros((size, size), dtype=np.float32)
    num_regions_actual = random.randint(2, 4)
    details_per_region_actual = random.randint(8, 15)
    foam_elements_actual = random.randint(30, 60)
    for _ in range(num_regions_actual):
        region_type = random.choice(['mixed_shapes', 'foam_area'])
        r_start = random.randint(0, size // 2)
        c_start = random.randint(0, size // 2)
        r_end = random.randint(r_start + size // 4, size -1)
        c_end = random.randint(c_start + size // 4, size -1)
        region_value_base = random.uniform(0.1, 0.7)
        if region_type == 'mixed_shapes':
            for _ in range(details_per_region_actual):
                shape_type = random.choice(['ellipse', 'rectangle', 'polygon'])
                value = region_value_base + random.uniform(-0.1, 0.5)
                r_c = random.randint(r_start, r_end)
                c_c = random.randint(c_start, c_end)
                if shape_type == 'ellipse':
                    r_rad = random.randint(max(1,size // 32), size // 10)
                    c_rad = random.randint(max(1,size // 32), size // 10)
                    orientation = random.uniform(0, np.pi)
                    rr, cc = ellipse(r_c, c_c, r_rad, c_rad, shape=(size, size), rotation=orientation)
                    phantom[rr, cc] = value
                elif shape_type == 'rectangle':
                    width = random.randint(size // 20, size // 8)
                    height = random.randint(size // 20, size // 8)
                    s_r, s_c = max(0, r_c - height//2), max(0, c_c - width//2)
                    e_r, e_c = min(size-1, s_r + height), min(size-1, s_c + width)
                    if e_r > s_r and e_c > s_c:
                         rr, cc = rectangle((s_r, s_c), end=(e_r, e_c), shape=(size, size))
                         phantom[rr, cc] = value
                elif shape_type == 'polygon':
                    num_v = random.randint(3,5)
                    verts_r = np.clip(r_c + np.random.randint(-size//10, size//10, num_v), 0, size-1)
                    verts_c = np.clip(c_c + np.random.randint(-size//10, size//10, num_v), 0, size-1)
                    if len(verts_r) >=3 :
                        try:
                            rr, cc = polygon(verts_r, verts_c, shape=(size,size))
                            phantom[rr,cc] = value
                        except: pass
        elif region_type == 'foam_area':
            rr_foam_bg, cc_foam_bg = rectangle((r_start, c_start), end=(r_end, c_end), shape=(size,size))
            phantom[rr_foam_bg, cc_foam_bg] = np.maximum(phantom[rr_foam_bg, cc_foam_bg], region_value_base * 0.5)
            for _ in range(foam_elements_actual):
                r_foam = random.randint(r_start, r_end)
                c_foam = random.randint(c_start, c_end)
                rad_foam = random.randint(max(1,size // 64), size // 20)
                val_foam = region_value_base + random.uniform(-0.25, 0.25)
                rr, cc = disk((r_foam, c_foam), rad_foam, shape=(size,size))
                phantom[rr,cc] = np.clip(val_foam, 0, 1.5)
    for _ in range(details_per_region_actual * 2):
        r_detail = random.randint(0, size-1)
        c_detail = random.randint(0, size-1)
        rad_detail = random.randint(1, max(2, size // 40))
        val_detail = random.uniform(1.0, 1.5)
        rr, cc = disk((r_detail, c_detail), rad_detail, shape=(size,size))
        phantom[rr,cc] = val_detail
    return np.clip(phantom, 0, 1.5)

# --- 2. CT Simulation ---
def simulate_projections(phantom_2d, n_projections=N_PROJECTIONS):
    angles = np.linspace(0., 180., n_projections, endpoint=False)
    sinogram = radon(phantom_2d, theta=angles, circle=True)
    return sinogram, angles

def add_noise_ct(sinogram, noise_type='gaussian', noise_level_gaussian=0.1, noise_level_poisson_lam=30): # Renamed from add_noise
    noisy_sinogram = sinogram.copy()
    max_sino_val = np.max(sinogram)
    if max_sino_val == 0: max_sino_val = 1.0 
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level_gaussian * max_sino_val, sinogram.shape)
        noisy_sinogram += noise
    elif noise_type == 'poisson':
        scaled_sinogram = (noisy_sinogram / np.maximum(max_sino_val, 1e-9)) * noise_level_poisson_lam # ensure max_sino_val > 0
        noisy_sinogram = np.random.poisson(np.maximum(0, scaled_sinogram)).astype(np.float32)
        noisy_sinogram = (noisy_sinogram / noise_level_poisson_lam) * max_sino_val
    else:
        raise ValueError("Unknown CT noise type. Choose 'gaussian' or 'poisson'.")
    return np.clip(noisy_sinogram, 0, None)

def reconstruct_fbp(sinogram, angles):
    reconstruction_fbp = iradon(sinogram, theta=angles, circle=True)
    if reconstruction_fbp.shape[0] != PHANTOM_SIZE_XY or reconstruction_fbp.shape[1] != PHANTOM_SIZE_XY:
        reconstruction_fbp = resize(reconstruction_fbp, (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY), 
                                    anti_aliasing=True, mode='reflect')
    return reconstruction_fbp.astype(np.float32)

# --- 3. CNN Architectures (PyTorch) --- (UNet, DnCNN, REDNet remain unchanged)
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
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

class REDNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=5, num_features=64):
        super(REDNet, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU(inplace=True)
        self.conv_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=True))
        for _ in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
        for _ in range(num_layers - 1):
            self.deconv_layers.append(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
        self.deconv_layers.append(nn.ConvTranspose2d(num_features, out_channels, kernel_size=3, padding=1, bias=True))
    def forward(self, x):
        encoder_outputs = []
        h = x
        for i in range(self.num_layers):
            h = self.conv_layers[i](h)
            h = self.relu(h)
            if i < self.num_layers - 1:
                encoder_outputs.append(h)
        for i in range(self.num_layers):
            h = self.deconv_layers[i](h)
            if i < self.num_layers - 1:
                skip_val = encoder_outputs[self.num_layers - 2 - i]
                h = h + skip_val
                h = self.relu(h)
        return h

# --- 4. Noise2Inverse Data Handling and Training ---
# For CT
class Noise2InverseDatasetCT(Dataset):
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
                for recon in sub_reconstructions_raw ]
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
                else: raise ValueError(f"Unknown strategy: {self.strategy}")
                pairs.append((input_recon, target_recon))
        return pairs
    def __len__(self): return len(self.training_pairs)
    def __getitem__(self, idx):
        input_recon, target_recon = self.training_pairs[idx]
        # Normalize input and target reconstruction images for the network
        input_norm = normalize_phantom_for_mri(input_recon) # Using mri normalizer for general 0-1
        target_norm = normalize_phantom_for_mri(target_recon)
        input_tensor = torch.from_numpy(input_norm).unsqueeze(0)
        target_tensor = torch.from_numpy(target_norm).unsqueeze(0)
        return input_tensor, target_tensor

# For MRI
class Noise2InverseDatasetMRI(Dataset):
    def __init__(self, list_of_acquired_kspace_paths, kspace_shape):
        self.list_of_acquired_kspace_paths = list_of_acquired_kspace_paths
        self.kspace_shape = kspace_shape # e.g. (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY)

    def __len__(self):
        return len(self.list_of_acquired_kspace_paths)

    def __getitem__(self, idx):
        acquired_kspace_noisy_sparse = np.load(self.list_of_acquired_kspace_paths[idx]) # Complex valued

        # Get indices of acquired points
        acquired_indices_r, acquired_indices_c = np.where(acquired_kspace_noisy_sparse != 0)
        num_acquired = len(acquired_indices_r)

        if num_acquired < 2: # Need at least 2 points to split
            # Fallback: return reconstructed image from all acquired points as both input and target
            # This isn't ideal for N2I but prevents crashing with too few points.
            # A better handling might be to skip such data during _prepare_training_pairs if done there.
            img = get_image_from_kspace(acquired_kspace_noisy_sparse)
            img_norm = normalize_phantom_for_mri(img)
            img_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
            return img_tensor, img_tensor

        # Shuffle and split acquired indices into two halves
        shuffled_order = np.random.permutation(num_acquired)
        split_point = num_acquired // 2
        
        indices_half1_shuffled = shuffled_order[:split_point]
        indices_half2_shuffled = shuffled_order[split_point:]

        k_half1 = np.zeros(self.kspace_shape, dtype=np.complex64)
        k_half2 = np.zeros(self.kspace_shape, dtype=np.complex64)

        # Populate k_half1
        r1 = acquired_indices_r[indices_half1_shuffled]
        c1 = acquired_indices_c[indices_half1_shuffled]
        k_half1[r1, c1] = acquired_kspace_noisy_sparse[r1, c1]
        
        # Populate k_half2
        r2 = acquired_indices_r[indices_half2_shuffled]
        c2 = acquired_indices_c[indices_half2_shuffled]
        k_half2[r2, c2] = acquired_kspace_noisy_sparse[r2, c2]

        image_half1 = get_image_from_kspace(k_half1)
        image_half2 = get_image_from_kspace(k_half2)

        # Normalize images before converting to tensor
        image_half1_norm = normalize_phantom_for_mri(image_half1)
        image_half2_norm = normalize_phantom_for_mri(image_half2)

        image_half1_tensor = torch.from_numpy(image_half1_norm).unsqueeze(0).float()
        image_half2_tensor = torch.from_numpy(image_half2_norm).unsqueeze(0).float()
        
        return image_half1_tensor, image_half2_tensor


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

# --- 5. Evaluation Metrics --- (Unchanged)
def evaluate_denoising_metrics(denoised_img, ground_truth_img, noisy_img):
    def normalize_for_metrics(img):
        if img is None: return None
        # Assumes ground_truth_img (0-1) is the reference for normalization range if needed
        # For N2I, inputs are already somewhat normalized by dataset loader
        img_min, img_max = np.min(img), np.max(img)
        return (img - img_min) / (img_max - img_min) if img_max > img_min else img
    
    # Ground truth is assumed to be clean and well-scaled (e.g., 0-1)
    # Noisy and Denoised should be compared against this GT.
    # Normalization for metrics should ensure all images are in a comparable range (e.g. 0-1)
    # for PSNR/SSIM data_range=1.0
    
    # If gt_norm is already 0-1, no need to normalize it again.
    gt_norm = ground_truth_img # Assuming GT is already appropriately scaled (e.g. 0-1)
    
    denoised_norm = normalize_for_metrics(denoised_img)
    noisy_norm = normalize_for_metrics(noisy_img)


    if gt_norm is None or noisy_norm is None:
        psnr_noisy, ssim_noisy = np.nan, np.nan
    else:
        # Ensure shapes match, especially if GT is original phantom and noisy is reconstructed
        if gt_norm.shape != noisy_norm.shape:
             noisy_norm = resize(noisy_norm, gt_norm.shape, anti_aliasing=True, mode='reflect')

        psnr_noisy = psnr(gt_norm, noisy_norm, data_range=1.0) # Assuming data range 0-1 after normalization
        ssim_noisy = ssim(gt_norm, noisy_norm, data_range=1.0, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    print(f"Noisy Image: PSNR={psnr_noisy:.2f} dB, SSIM={ssim_noisy:.4f}")

    if gt_norm is None or denoised_norm is None:
        psnr_denoised, ssim_denoised = np.nan, np.nan
    else:
        if gt_norm.shape != denoised_norm.shape:
             denoised_norm = resize(denoised_norm, gt_norm.shape, anti_aliasing=True, mode='reflect')
        psnr_denoised = psnr(gt_norm, denoised_norm, data_range=1.0)
        ssim_denoised = ssim(gt_norm, denoised_norm, data_range=1.0, channel_axis=None, win_size=min(7, gt_norm.shape[0], gt_norm.shape[1]))
    print(f"Denoised Image: PSNR={psnr_denoised:.2f} dB, SSIM={ssim_denoised:.4f}")
    
    return psnr_denoised, ssim_denoised, psnr_noisy, ssim_noisy

# --- Helper function for argparse --- (Unchanged)
def str_to_bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Main Execution with Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise2Inverse Experiment Pipeline for CT and MRI")
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate_data', 'train_unet', 'train_dncnn', 'train_rednet', 'evaluate', 'full_run', 'visualize_phantom'],
                        help='Action to perform.')
    parser.add_argument('--modality', type=str, default='ct', choices=['ct', 'mri'], help='Imaging modality to use.')
    parser.add_argument('--num_train_phantoms', type=int, default=20, help='Number of phantoms for training.')
    parser.add_argument('--num_eval_phantoms', type=int, default=3, help='Number of phantoms for evaluation.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    # CT specific args (could be conditionally shown or have defaults ignored for MRI)
    parser.add_argument('--k_splits', type=int, default=K_SPLITS_NOISE2INVERSE, help='K splits for Noise2Inverse dataset (CT).')
    parser.add_argument('--noise_type_arg', type=str, default=NOISE_TYPE, choices=['gaussian', 'poisson'], help='Type of noise to add (CT).')
    # MRI specific args (could add more like acceleration, noise level for MRI if not using fixed constants)
    parser.add_argument('--mri_accel', type=float, default=MRI_ACCELERATION_FACTOR, help='MRI K-space acceleration factor.')
    parser.add_argument('--mri_noise_level', type=float, default=MRI_KSPACE_NOISE_PERCENT, help='MRI K-space noise level (percent).')

    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--plot_data', type=str_to_bool, nargs='?', const=True, default=True, help='Plot generated data samples and evaluation results.')
    parser.add_argument('--model_load_path', type=str, default='trained_models', help='Directory to save/load trained models.')
    parser.add_argument('--data_save_path', type=str, default='generated_data', help='Directory to save/load generated data.')

    args = parser.parse_args()

    os.makedirs(args.data_save_path, exist_ok=True)
    os.makedirs(args.model_load_path, exist_ok=True)

    # --- Data Generation ---
    if args.action in ['generate_data', 'full_run'] or \
       (args.action in ['train_unet', 'train_dncnn', 'train_rednet'] and not os.listdir(args.data_save_path)): # Auto-generate if data path is empty for training
        
        if args.modality == 'ct':
            print("--- Generating CT Data ---")
            original_phantoms_2d_train_ct = []
            training_noisy_sinograms_ct = []
            training_angles_ct = None
            for i in range(args.num_train_phantoms):
                p_2d = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
                original_phantoms_2d_train_ct.append(p_2d)
                np.save(os.path.join(args.data_save_path, f"train_ct_phantom_2d_{i}.npy"), p_2d)
                sinogram, angles_current = simulate_projections(p_2d, N_PROJECTIONS)
                if training_angles_ct is None: training_angles_ct = angles_current
                noisy_sinogram = add_noise_ct(sinogram, args.noise_type_arg, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
                training_noisy_sinograms_ct.append(noisy_sinogram)
                np.save(os.path.join(args.data_save_path, f"train_ct_sino_noisy_{i}.npy"), noisy_sinogram)
            if training_angles_ct is not None: np.save(os.path.join(args.data_save_path, f"train_ct_angles.npy"), training_angles_ct)
            print(f"Generated {args.num_train_phantoms} CT training phantoms and noisy sinograms.")

            for i in range(args.num_eval_phantoms):
                p_2d_eval_gt = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
                sino_gt, angles_curr_eval = simulate_projections(p_2d_eval_gt, N_PROJECTIONS)
                sino_noisy = add_noise_ct(sino_gt, args.noise_type_arg, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
                fbp_gt_ct = reconstruct_fbp(sino_gt, angles_curr_eval) # This is recon from clean sino
                fbp_noisy_ct = reconstruct_fbp(sino_noisy, angles_curr_eval)
                np.save(os.path.join(args.data_save_path, f"eval_ct_phantom_gt_{i}.npy"), p_2d_eval_gt) # True GT phantom
                # np.save(os.path.join(args.data_save_path, f"eval_ct_fbp_gt_{i}.npy"), fbp_gt_ct) # FBP of clean
                np.save(os.path.join(args.data_save_path, f"eval_ct_fbp_noisy_{i}.npy"), fbp_noisy_ct) # FBP of noisy
                if i == 0 and angles_curr_eval is not None: np.save(os.path.join(args.data_save_path, f"eval_ct_angles.npy"), angles_curr_eval)
            print(f"Generated {args.num_eval_phantoms} CT evaluation phantoms and FBP reconstructions.")

        elif args.modality == 'mri':
            print("--- Generating MRI Data ---")
            # Training data
            for i in range(args.num_train_phantoms):
                phantom_raw = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
                phantom_gt_mri = normalize_phantom_for_mri(phantom_raw)
                
                kspace_gt = get_kspace_from_image(phantom_gt_mri)
                mask = generate_variable_density_mask(kspace_gt.shape, args.mri_accel, MRI_CENTRAL_FRACTION_XY)
                acquired_kspace_clean = apply_kspace_mask(kspace_gt, mask)
                acquired_kspace_noisy = add_kspace_noise_to_acquired(acquired_kspace_clean, mask, args.mri_noise_level)
                
                np.save(os.path.join(args.data_save_path, f"train_mri_phantom_gt_{i}.npy"), phantom_gt_mri)
                np.save(os.path.join(args.data_save_path, f"train_mri_acquired_kspace_noisy_{i}.npy"), acquired_kspace_noisy)
                if i == 0: np.save(os.path.join(args.data_save_path, f"train_mri_kspace_mask_default.npy"), mask) # Save one mask
            print(f"Generated {args.num_train_phantoms} MRI training phantoms and noisy acquired k-spaces.")

            # Evaluation data
            for i in range(args.num_eval_phantoms):
                phantom_raw_eval = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
                phantom_gt_mri_eval = normalize_phantom_for_mri(phantom_raw_eval)

                kspace_gt_eval = get_kspace_from_image(phantom_gt_mri_eval)
                mask_eval = generate_variable_density_mask(kspace_gt_eval.shape, args.mri_accel, MRI_CENTRAL_FRACTION_XY)
                acquired_kspace_clean_eval = apply_kspace_mask(kspace_gt_eval, mask_eval)
                acquired_kspace_noisy_eval = add_kspace_noise_to_acquired(acquired_kspace_clean_eval, mask_eval, args.mri_noise_level)
                
                # This is the input to N2I during inference for MRI
                recon_noisy_aliased_mri = get_image_from_kspace(acquired_kspace_noisy_eval) 
                
                np.save(os.path.join(args.data_save_path, f"eval_mri_phantom_gt_{i}.npy"), phantom_gt_mri_eval) # True GT
                np.save(os.path.join(args.data_save_path, f"eval_mri_recon_noisy_aliased_{i}.npy"), recon_noisy_aliased_mri) # N2I input
                np.save(os.path.join(args.data_save_path, f"eval_mri_kspace_mask_{i}.npy"), mask_eval)
                # For reference, also save the noisy acquired k-space used for this recon
                np.save(os.path.join(args.data_save_path, f"eval_mri_acquired_kspace_noisy_{i}.npy"), acquired_kspace_noisy_eval)


                if args.plot_data and i < 1: # Plot for the first eval MRI phantom
                    fig_mri_data, axes_mri = plt.subplots(1, 4, figsize=(20, 5))
                    axes_mri[0].imshow(phantom_gt_mri_eval, cmap='gray', vmin=0, vmax=1); axes_mri[0].set_title(f"MRI GT Phantom {i+1}"); axes_mri[0].axis('off')
                    axes_mri[1].imshow(np.log(1 + np.abs(acquired_kspace_noisy_eval)), cmap='viridis'); axes_mri[1].set_title("Acquired K-space (log)"); axes_mri[1].axis('off')
                    axes_mri[2].imshow(mask_eval, cmap='gray'); axes_mri[2].set_title("K-space Mask"); axes_mri[2].axis('off')
                    axes_mri[3].imshow(recon_noisy_aliased_mri, cmap='gray', vmin=0, vmax=np.max(recon_noisy_aliased_mri)); axes_mri[3].set_title("Noisy/Aliased MRI Recon"); axes_mri[3].axis('off')
                    plt.tight_layout()
                    plt.show()
            print(f"Generated {args.num_eval_phantoms} MRI evaluation phantoms and related data.")
        
        if args.action == 'generate_data':
            print(f"Data generation complete for {args.modality}. Files saved in {args.data_save_path}")

    if args.action == 'visualize_phantom':
        print(f"Visualizing a sample 2D phantom (intended for {args.modality})...")
        phantom_2d_viz_raw = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
        if args.modality == 'mri':
            phantom_2d_viz = normalize_phantom_for_mri(phantom_2d_viz_raw)
            vmax_plot = 1.0
        else: # CT
            phantom_2d_viz = phantom_2d_viz_raw
            vmax_plot = np.max(phantom_2d_viz) if np.max(phantom_2d_viz) > 0 else 1.0
        
        plt.figure(figsize=(7,7))
        plt.imshow(phantom_2d_viz, cmap='gray', vmin=0, vmax=vmax_plot)
        plt.title(f"Dense & Varied 2D Phantom Sample ({args.modality})")
        plt.colorbar()
        plt.show()

        if args.modality == 'mri' and args.plot_data:
            kspace_viz = get_kspace_from_image(phantom_2d_viz)
            mask_viz = generate_variable_density_mask(kspace_viz.shape, args.mri_accel, MRI_CENTRAL_FRACTION_XY)
            acquired_k_viz = apply_kspace_mask(kspace_viz, mask_viz)
            noisy_acquired_k_viz = add_kspace_noise_to_acquired(acquired_k_viz, mask_viz, args.mri_noise_level)
            recon_aliased_viz = get_image_from_kspace(noisy_acquired_k_viz)

            fig_mri_viz, axes_mri_v = plt.subplots(1,4, figsize=(20,5))
            axes_mri_v[0].imshow(phantom_2d_viz, cmap='gray', vmin=0, vmax=1); axes_mri_v[0].set_title("Normalized Phantom"); axes_mri_v[0].axis('off')
            axes_mri_v[1].imshow(mask_viz, cmap='gray'); axes_mri_v[1].set_title("K-space Mask"); axes_mri_v[1].axis('off')
            axes_mri_v[2].imshow(np.log(1+np.abs(noisy_acquired_k_viz)), cmap='viridis'); axes_mri_v[2].set_title("Noisy Acquired K (log)"); axes_mri_v[2].axis('off')
            axes_mri_v[3].imshow(recon_aliased_viz, cmap='gray'); axes_mri_v[3].set_title("Recon (Noisy, Aliased)"); axes_mri_v[3].axis('off')
            plt.tight_layout()
            plt.show()


    # --- Model Training ---
    if args.action in ['train_unet', 'train_dncnn', 'train_rednet', 'full_run']:
        n2i_dataloader = None
        if args.modality == 'ct':
            print(f"Loading CT training data from {args.data_save_path}...")
            training_noisy_sinograms_ct = []
            try:
                for i in range(args.num_train_phantoms):
                    training_noisy_sinograms_ct.append(np.load(os.path.join(args.data_save_path, f"train_ct_sino_noisy_{i}.npy")))
                training_angles_ct = np.load(os.path.join(args.data_save_path, f"train_ct_angles.npy"))
                if not training_noisy_sinograms_ct or training_angles_ct is None: raise FileNotFoundError # Check if lists are empty
            except FileNotFoundError:
                print("Error: CT Training sinogram/angle data not found. Run --action generate_data --modality ct first."); exit()
            
            n2i_dataset_ct = Noise2InverseDatasetCT(training_noisy_sinograms_ct, training_angles_ct, 
                                               k_splits=args.k_splits, recon_size=PHANTOM_SIZE_XY)
            if len(n2i_dataset_ct) == 0: print("CT Dataset is empty."); exit()
            n2i_dataloader = DataLoader(n2i_dataset_ct, batch_size=args.batch_size, shuffle=True, num_workers=0)
            print(f"CT Noise2Inverse dataset size: {len(n2i_dataset_ct)} pairs.")

        elif args.modality == 'mri':
            print(f"Loading MRI training data from {args.data_save_path}...")
            training_acquired_kspace_paths_mri = []
            for i in range(args.num_train_phantoms):
                fpath = os.path.join(args.data_save_path, f"train_mri_acquired_kspace_noisy_{i}.npy")
                if os.path.exists(fpath):
                    training_acquired_kspace_paths_mri.append(fpath)
                else:
                    print(f"Warning: Missing {fpath}")
            
            if not training_acquired_kspace_paths_mri:
                print("Error: MRI Training k-space data not found. Run --action generate_data --modality mri first."); exit()

            n2i_dataset_mri = Noise2InverseDatasetMRI(training_acquired_kspace_paths_mri, kspace_shape=(PHANTOM_SIZE_XY, PHANTOM_SIZE_XY))
            if len(n2i_dataset_mri) == 0: print("MRI Dataset is empty."); exit()
            n2i_dataloader = DataLoader(n2i_dataset_mri, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity
            print(f"MRI Noise2Inverse dataset size: {len(n2i_dataset_mri)} pairs.")

        if n2i_dataloader is None:
            print("Error: Dataloader not initialized. Modality might be missing or data generation failed."); exit()

        criterion = nn.MSELoss()
        model_name_suffix = f"_{args.modality.upper()}" # e.g. U-Net_CT

        if args.action in ['train_unet', 'full_run']:
            unet_model = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_unet = optim.Adam(unet_model.parameters(), lr=args.lr)
            train_model(unet_model, n2i_dataloader, criterion, optimizer_unet, args.epochs, f"U-Net{model_name_suffix}", save_path=args.model_load_path)
        if args.action in ['train_dncnn', 'full_run']:
            dncnn_model = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=args.lr)
            train_model(dncnn_model, n2i_dataloader, criterion, optimizer_dncnn, args.epochs, f"DnCNN{model_name_suffix}", save_path=args.model_load_path)
        if args.action in ['train_rednet', 'full_run']:
            rednet_model = REDNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_rednet = optim.Adam(rednet_model.parameters(), lr=args.lr)
            train_model(rednet_model, n2i_dataloader, criterion, optimizer_rednet, args.epochs, f"REDNet{model_name_suffix}", save_path=args.model_load_path)
        
        # Combined loss plot for models trained in this session for the current modality
        if args.plot_data:
            plt.figure(figsize=(10,5))
            legend_added_plot = False
            for model_key in training_losses_history:
                if model_key.endswith(model_name_suffix): # Plot only for current modality
                    plt.plot(training_losses_history[model_key], label=f"{model_key} Training Loss")
                    legend_added_plot = True
            if legend_added_plot:
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title(f"Comparative Training Convergence ({args.modality.upper()})"); plt.legend(); plt.grid(True)
                plt.show()

    # --- Model Evaluation ---
    if args.action in ['evaluate', 'full_run']:
        print(f"\n--- Evaluating Models for Modality: {args.modality.upper()} ---")
        
        eval_gt_phantoms = [] # Will store actual GT images (normalized phantoms for MRI, raw for CT to be used with FBP)
        eval_noisy_inputs_for_n2i = [] # Store FBP noisy for CT, or Recon Aliased for MRI
        
        if args.modality == 'ct':
            for i in range(args.num_eval_phantoms):
                try:
                    # For CT, GT for metrics is the original phantom before any reconstruction if available
                    # For N2I evaluation, the input is noisy FBP, compare N2I output to original phantom
                    gt_phantom = np.load(os.path.join(args.data_save_path, f"eval_ct_phantom_gt_{i}.npy"))
                    noisy_fbp = np.load(os.path.join(args.data_save_path, f"eval_ct_fbp_noisy_{i}.npy"))
                    eval_gt_phantoms.append(gt_phantom)
                    eval_noisy_inputs_for_n2i.append(noisy_fbp)
                except FileNotFoundError:
                    print(f"Error: CT Evaluation data for phantom {i} not found. Run generate_data first."); exit()
        elif args.modality == 'mri':
            for i in range(args.num_eval_phantoms):
                try:
                    gt_phantom_mri = np.load(os.path.join(args.data_save_path, f"eval_mri_phantom_gt_{i}.npy")) # Already normalized 0-1
                    recon_noisy_aliased = np.load(os.path.join(args.data_save_path, f"eval_mri_recon_noisy_aliased_{i}.npy"))
                    eval_gt_phantoms.append(gt_phantom_mri)
                    eval_noisy_inputs_for_n2i.append(recon_noisy_aliased)
                except FileNotFoundError:
                    print(f"Error: MRI Evaluation data for phantom {i} not found. Run generate_data first."); exit()

        if not eval_gt_phantoms: print("No evaluation data loaded."); exit()

        unet_model_eval = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        dncnn_model_eval = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        rednet_model_eval = REDNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        models_to_eval = {'U-Net': unet_model_eval, 'DnCNN': dncnn_model_eval, 'REDNet': rednet_model_eval}
        loaded_models = {}

        for model_name, model_instance in models_to_eval.items():
            # Try loading modality specific model first, then generic
            model_sfx_name = f"{model_name}_{args.modality.upper()}"
            model_filenames_to_try = [
                f"{model_sfx_name.lower().replace(' ', '_').replace('-', '_')}_trained.pth", # Modality specific
                f"{model_name.lower().replace(' ', '_').replace('-', '_')}_trained.pth" # Generic
            ]
            loaded_successfully = False
            for fname in model_filenames_to_try:
                fpath = os.path.join(args.model_load_path, fname)
                if os.path.exists(fpath):
                    try:
                        model_instance.load_state_dict(torch.load(fpath, map_location=DEVICE))
                        model_instance.to(DEVICE).eval()
                        loaded_models[model_name] = model_instance
                        print(f"{model_name} ({args.modality}) model loaded from {fpath}")
                        loaded_successfully = True
                        break 
                    except Exception as e:
                        print(f"Error loading {fname}: {str(e)}")
            if not loaded_successfully:
                print(f"{model_name} model not found or failed to load for {args.modality}. Skipping its evaluation.")
        
        for i in range(len(eval_gt_phantoms)):
            current_gt_phantom = eval_gt_phantoms[i]
            current_noisy_input = eval_noisy_inputs_for_n2i[i] # This is the image N2I will denoise

            print(f"\n--- Evaluating on Phantom {i+1} ({args.modality.upper()}) ---")
            
            # Plot GT and Noisy Input once
            if args.plot_data:
                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(current_gt_phantom, cmap='gray', vmin=0, vmax=np.max(current_gt_phantom) if np.max(current_gt_phantom)>0 else 1)
                plt.title(f"Ground Truth Phantom {i+1}"); plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(current_noisy_input, cmap='gray', vmin=0, vmax=np.max(current_noisy_input) if np.max(current_noisy_input)>0 else 1)
                plt.title(f"Noisy Input for N2I ({args.modality.upper()})"); plt.axis('off')
                plt.show()
                if args.modality == 'mri': # Show k-space mask for MRI
                    try:
                        mask_eval_mri = np.load(os.path.join(args.data_save_path, f"eval_mri_kspace_mask_{i}.npy"))
                        plt.figure(figsize=(5,5))
                        plt.imshow(mask_eval_mri, cmap='gray'); plt.title(f"K-space Mask Phantom {i+1}"); plt.axis('off')
                        plt.show()
                    except: pass


            denoised_outputs = {}
            residual_images = {}

            with torch.no_grad():
                for model_name, model_eval_instance in loaded_models.items():
                    # For N2I, the input is the single noisy image (FBP for CT, Aliased Recon for MRI)
                    # The N2I *training* uses splits, but inference is usually on the full noisy data.
                    input_tensor = torch.from_numpy(normalize_phantom_for_mri(current_noisy_input)).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                    denoised_outputs[model_name] = model_eval_instance(input_tensor).squeeze().cpu().numpy()
                    residual_images[model_name] = denoised_outputs[model_name] - current_gt_phantom # Residual against true GT
                    
                    print(f"\n{model_name} ({args.modality.upper()}) N2I Evaluation Results:")
                    evaluate_denoising_metrics(denoised_outputs[model_name], current_gt_phantom, current_noisy_input)

            if args.plot_data:
                # Individual plots for each model
                for model_name in loaded_models.keys():
                    if model_name in denoised_outputs:
                        fig_model, ax_model = plt.subplots(1, 3, figsize=(15, 5))
                        common_kwargs_eval = {'cmap': 'gray', 'vmin': 0, 'vmax': np.max(current_gt_phantom) if np.max(current_gt_phantom)>0 else 1}

                        ax_model[0].imshow(current_gt_phantom, **common_kwargs_eval); ax_model[0].set_title("Ground Truth"); ax_model[0].axis('off')
                        ax_model[1].imshow(current_noisy_input, **common_kwargs_eval); ax_model[1].set_title("Noisy Input"); ax_model[1].axis('off')
                        ax_model[2].imshow(denoised_outputs[model_name], **common_kwargs_eval); ax_model[2].set_title(f"{model_name} Denoised"); ax_model[2].axis('off')
                        plt.tight_layout(); plt.suptitle(f"{model_name} Denoising Results ({args.modality.upper()})", fontsize=14); plt.subplots_adjust(top=0.85); plt.show()
                
                # Comparative Residual plot
                num_residuals_to_plot_eval = len(residual_images)
                if num_residuals_to_plot_eval > 0:
                    plt.figure(figsize=(5 * num_residuals_to_plot_eval, 4))
                    plot_idx_eval = 1
                    # Determine a common scale for residual plots if desired, e.g., based on GT range
                    res_vmax_plot = 0.3 * (np.max(current_gt_phantom) if np.max(current_gt_phantom) > 0 else 1.0)
                    res_vmin_plot = -res_vmax_plot
                    
                    for model_name_res, res_img in residual_images.items():
                        plt.subplot(1, num_residuals_to_plot_eval, plot_idx_eval)
                        plt.imshow(res_img, cmap='coolwarm', vmin=res_vmin_plot, vmax=res_vmax_plot)
                        plt.title(f"Residual ({model_name_res} - GT)"); plt.axis('off'); plt.colorbar()
                        plot_idx_eval +=1
                    plt.tight_layout(); plt.suptitle(f"Comparative Residuals ({args.modality.upper()})", fontsize=14); plt.subplots_adjust(top=0.85 if num_residuals_to_plot_eval > 1 else 0.75); plt.show()

    print("\nScript finished.")
    if args.action == 'generate_data': print(f"Data generation complete for {args.modality}. Files saved in {args.data_save_path}")
    print("If denoising is still suboptimal: try more epochs, more training phantoms, tune learning rate/batch size, or adjust K-splits (CT) / N2I-MRI data splitting.")