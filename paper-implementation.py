import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.transform import radon, iradon, resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.draw import ellipse, rectangle, polygon, disk # Added disk for foam
import matplotlib.pyplot as plt
import random
import argparse
import os
from datetime import datetime

# --- Configuration ---
PHANTOM_SIZE_XY = 128
N_PROJECTIONS = 180
NOISE_TYPE = 'poisson'
NOISE_LEVEL_GAUSSIAN = 0.1
NOISE_LEVEL_POISSON_LAM = 30
MODEL_INPUT_CHANNELS = 1
MODEL_OUTPUT_CHANNELS = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 4
K_SPLITS_NOISE2INVERSE = 2

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

# Global dictionary to store losses for combined plot
training_losses_history = {}

# --- 1. Dense & Varied 2D Phantom Generation ---
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

def add_noise(sinogram, noise_type='gaussian', noise_level_gaussian=0.1, noise_level_poisson_lam=30):
    noisy_sinogram = sinogram.copy()
    max_sino_val = np.max(sinogram)
    if max_sino_val == 0: max_sino_val = 1.0 
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level_gaussian * max_sino_val, sinogram.shape)
        noisy_sinogram += noise
    elif noise_type == 'poisson':
        scaled_sinogram = (noisy_sinogram / max_sino_val) * noise_level_poisson_lam
        noisy_sinogram = np.random.poisson(np.maximum(0, scaled_sinogram)).astype(np.float32)
        noisy_sinogram = (noisy_sinogram / noise_level_poisson_lam) * max_sino_val
    else:
        raise ValueError("Unknown noise type. Choose 'gaussian' or 'poisson'.")
    return np.clip(noisy_sinogram, 0, None)

def reconstruct_fbp(sinogram, angles):
    reconstruction_fbp = iradon(sinogram, theta=angles, circle=True)
    if reconstruction_fbp.shape[0] != PHANTOM_SIZE_XY or reconstruction_fbp.shape[1] != PHANTOM_SIZE_XY:
        reconstruction_fbp = resize(reconstruction_fbp, (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY), 
                                    anti_aliasing=True, mode='reflect')
    return reconstruction_fbp.astype(np.float32)

# --- 3. CNN Architectures (PyTorch) ---
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

# --- 5. Evaluation Metrics ---
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

# --- Helper function for argparse ---
def str_to_bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Main Execution with Argparse ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise2Inverse Experiment Pipeline")
    parser.add_argument('--action', type=str, required=True,
                        choices=['generate_data', 'train_unet', 'train_dncnn', 'train_rednet', 'evaluate', 'full_run', 'visualize_phantom'],
                        help='Action to perform.')
    parser.add_argument('--num_train_phantoms', type=int, default=20, help='Number of phantoms for training.')
    parser.add_argument('--num_eval_phantoms', type=int, default=3, help='Number of phantoms for evaluation.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--k_splits', type=int, default=K_SPLITS_NOISE2INVERSE, help='K splits for Noise2Inverse dataset.')
    parser.add_argument('--noise_type_arg', type=str, default=NOISE_TYPE, choices=['gaussian', 'poisson'], help='Type of noise to add.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate.')
    parser.add_argument('--plot_data', type=str_to_bool, nargs='?', const=True, default=True, help='Plot generated data samples and evaluation results.')
    parser.add_argument('--model_load_path', type=str, default='trained_models', help='Directory to save/load trained models.')
    parser.add_argument('--data_save_path', type=str, default='generated_data', help='Directory to save/load generated data.')

    args = parser.parse_args()

    os.makedirs(args.data_save_path, exist_ok=True)
    os.makedirs(args.model_load_path, exist_ok=True)

    original_phantoms_2d_train = []
    training_noisy_sinograms = []
    training_angles = None
    
    eval_phantoms_gt_2d = []
    eval_sinos_gt_all = [] 
    eval_sinos_noisy_all = [] 
    eval_fbps_gt = []
    eval_fbps_noisy = []
    eval_angles_list = []

    if args.action in ['generate_data', 'full_run', 'train_unet', 'train_dncnn', 'train_rednet']:
        print("1. Generating training phantom data...")
        for i in range(args.num_train_phantoms):
            phantom_2d = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
            original_phantoms_2d_train.append(phantom_2d)
            if args.action == 'generate_data':
                 np.save(os.path.join(args.data_save_path, f"train_phantom_2d_{i}.npy"), phantom_2d)
            if args.plot_data and i < 2 :
                plt.figure(figsize=(6,6))
                plt.imshow(phantom_2d, cmap='gray', vmin=0, vmax=np.max(phantom_2d) if np.max(phantom_2d) > 0 else 1.0)
                plt.title(f"Generated 2D Training Phantom {i+1}")
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
                fig_ct, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(p_2d, cmap='gray'); axes[0].set_title(f"Orig Phantom {i+1}"); axes[0].axis('off')
                axes[1].imshow(sinogram, cmap='gray'); axes[1].set_title("Clean Sinogram"); axes[1].axis('off')
                axes[2].imshow(noisy_sinogram, cmap='gray'); axes[2].set_title(f"Noisy Sinogram ({args.noise_type_arg})"); axes[2].axis('off')
                plt.tight_layout()
                plt.show()
        if args.action == 'generate_data':
             print(f"Training data (phantoms, sinograms, angles) saved to {args.data_save_path}")

    if args.action in ['generate_data', 'full_run', 'evaluate']:
        print("1. Generating evaluation phantom data...")
        for i in range(args.num_eval_phantoms):
            phantom_2d_eval_gt = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
            eval_phantoms_gt_2d.append(phantom_2d_eval_gt)
            sino_gt, angles_curr_eval = simulate_projections(phantom_2d_eval_gt, N_PROJECTIONS)
            sino_noisy = add_noise(sino_gt, args.noise_type_arg, NOISE_LEVEL_GAUSSIAN, NOISE_LEVEL_POISSON_LAM)
            fbp_gt = reconstruct_fbp(sino_gt, angles_curr_eval)
            fbp_noisy = reconstruct_fbp(sino_noisy, angles_curr_eval)
            eval_sinos_gt_all.append(sino_gt)
            eval_sinos_noisy_all.append(sino_noisy)
            eval_fbps_gt.append(fbp_gt)
            eval_fbps_noisy.append(fbp_noisy)
            eval_angles_list.append(angles_curr_eval)
            if args.action == 'generate_data':
                np.save(os.path.join(args.data_save_path, f"eval_phantom_2d_gt_{i}.npy"), phantom_2d_eval_gt)
                np.save(os.path.join(args.data_save_path, f"eval_sino_gt_{i}.npy"), sino_gt)
                np.save(os.path.join(args.data_save_path, f"eval_sino_noisy_{i}.npy"), sino_noisy)
                np.save(os.path.join(args.data_save_path, f"eval_fbp_gt_{i}.npy"), fbp_gt)
                np.save(os.path.join(args.data_save_path, f"eval_fbp_noisy_{i}.npy"), fbp_noisy)
                if i == 0: np.save(os.path.join(args.data_save_path, f"eval_angles_eval.npy"), angles_curr_eval)
            if args.plot_data and i < 1:
                fig_eval_data, axes = plt.subplots(1, 4, figsize=(20, 5))
                axes[0].imshow(phantom_2d_eval_gt, cmap='gray'); axes[0].set_title(f"Eval GT Phantom {i+1}"); axes[0].axis('off')
                axes[1].imshow(sino_noisy, cmap='gray'); axes[1].set_title("Eval Noisy Sinogram"); axes[1].axis('off')
                axes[2].imshow(fbp_gt, cmap='gray'); axes[2].set_title("Eval GT FBP"); axes[2].axis('off')
                axes[3].imshow(fbp_noisy, cmap='gray'); axes[3].set_title("Eval Noisy FBP"); axes[3].axis('off')
                plt.tight_layout()
                plt.show()
        if args.action == 'generate_data':
            print(f"Evaluation data saved to {args.data_save_path}")

    if args.action == 'visualize_phantom':
        print("Visualizing a sample 2D dense & varied phantom...")
        phantom_2d_viz = generate_dense_varied_2d_phantom(PHANTOM_SIZE_XY)
        plt.figure(figsize=(7,7))
        plt.imshow(phantom_2d_viz, cmap='gray', vmin=0, vmax=np.max(phantom_2d_viz) if np.max(phantom_2d_viz) > 0 else 1.0)
        plt.title("Dense & Varied 2D Phantom Sample")
        plt.colorbar()
        plt.show()

    if args.action in ['train_unet', 'train_dncnn', 'train_rednet', 'full_run']:
        if not training_noisy_sinograms:
            print(f"Loading training data from {args.data_save_path}...")
            try:
                for i in range(args.num_train_phantoms):
                    training_noisy_sinograms.append(np.load(os.path.join(args.data_save_path, f"train_sino_noisy_{i}.npy")))
                training_angles = np.load(os.path.join(args.data_save_path, f"train_angles.npy"))
            except FileNotFoundError:
                print("Error: Training data not found. Run --action generate_data or full_run."); exit()
        
        print("3. Preparing Noise2Inverse DataLoader...")
        n2i_dataset = Noise2InverseDataset(training_noisy_sinograms, training_angles, 
                                           k_splits=args.k_splits, recon_size=PHANTOM_SIZE_XY)
        if len(n2i_dataset) == 0: print("Dataset is empty."); exit()
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
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("U-Net Training Convergence"); plt.legend(); plt.grid(True)
                plt.show()

        if args.action in ['train_dncnn', 'full_run']:
            print("4b. Initializing DnCNN model...")
            dncnn_model = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_dncnn = optim.Adam(dncnn_model.parameters(), lr=args.lr)
            dncnn_losses = train_model(dncnn_model, n2i_dataloader, criterion, optimizer_dncnn, args.epochs, "DnCNN", save_path=args.model_load_path)
            if args.plot_data:
                plt.figure(figsize=(10, 5))
                plt.plot(dncnn_losses, label="DnCNN Training Loss")
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("DnCNN Training Convergence"); plt.legend(); plt.grid(True)
                plt.show()
        
        if args.action in ['train_rednet', 'full_run']:
            print("4c. Initializing REDNet model...")
            rednet_model = REDNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
            optimizer_rednet = optim.Adam(rednet_model.parameters(), lr=args.lr)
            rednet_losses = train_model(rednet_model, n2i_dataloader, criterion, optimizer_rednet, args.epochs, "REDNet", save_path=args.model_load_path)
            if args.plot_data:
                plt.figure(figsize=(10, 5))
                plt.plot(rednet_losses, label="REDNet Training Loss")
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("REDNet Training Convergence"); plt.legend(); plt.grid(True)
                plt.show()
        
        if args.plot_data :
            legend_added = False
            plt.figure(figsize=(10,5))
            if 'U-Net' in training_losses_history:
                plt.plot(training_losses_history['U-Net'], label="U-Net Training Loss")
                legend_added = True
            if 'DnCNN' in training_losses_history:
                plt.plot(training_losses_history['DnCNN'], label="DnCNN Training Loss")
                legend_added = True
            if 'REDNet' in training_losses_history:
                plt.plot(training_losses_history['REDNet'], label="REDNet Training Loss")
                legend_added = True
            if legend_added:
                plt.xlabel("Epoch"); plt.ylabel("Avg MSE Loss"); plt.title("Comparative Training Convergence"); plt.legend(); plt.grid(True)
                plt.show()


    if args.action in ['evaluate', 'full_run']:
        if not eval_fbps_gt: 
            print(f"Loading evaluation data from {args.data_save_path}...")
            try:
                for i in range(args.num_eval_phantoms):
                    eval_fbps_gt.append(np.load(os.path.join(args.data_save_path, f"eval_fbp_gt_{i}.npy")))
                    eval_fbps_noisy.append(np.load(os.path.join(args.data_save_path, f"eval_fbp_noisy_{i}.npy")))
                    eval_sinos_noisy_all.append(np.load(os.path.join(args.data_save_path, f"eval_sino_noisy_{i}.npy")))
                    eval_angles_list.append(np.load(os.path.join(args.data_save_path, f"eval_angles_eval.npy"))) # Assuming angles are same for all eval
            except FileNotFoundError:
                print("Error: Evaluation FBP data not found. Run --action generate_data or full_run."); exit()

        print("\n6. Evaluating models with Noise2Inverse Inference Strategy...")
        unet_model_eval = UNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        dncnn_model_eval = DnCNN(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)
        rednet_model_eval = REDNet(MODEL_INPUT_CHANNELS, MODEL_OUTPUT_CHANNELS)

        unet_loaded, dncnn_loaded, rednet_loaded = False, False, False
        
        unet_filenames = ["u-net_trained.pth", "u_net_trained.pth", "unet_trained.pth"]
        dncnn_filenames = ["dncnn_trained.pth", "dncnn-trained.pth"]
        rednet_filename = "rednet_trained.pth"
        
        for unet_filename in unet_filenames:
            try:
                unet_path = os.path.join(args.model_load_path, unet_filename)
                if os.path.exists(unet_path):
                    unet_model_eval.load_state_dict(torch.load(unet_path, map_location=DEVICE))
                    unet_model_eval.to(DEVICE).eval()
                    unet_loaded = True
                    print(f"U-Net model loaded from {unet_path}")
                    break
            except Exception as e:
                print(f"Error loading {unet_filename}: {str(e)}")
        if not unet_loaded: print(f"U-Net model not found. Searched in {args.model_load_path} for: {', '.join(unet_filenames)}")
        
        for dncnn_filename in dncnn_filenames:
            try:
                dncnn_path = os.path.join(args.model_load_path, dncnn_filename)
                if os.path.exists(dncnn_path):
                    dncnn_model_eval.load_state_dict(torch.load(dncnn_path, map_location=DEVICE))
                    dncnn_model_eval.to(DEVICE).eval()
                    dncnn_loaded = True
                    print(f"DnCNN model loaded from {dncnn_path}")
                    break
            except Exception as e:
                print(f"Error loading {dncnn_filename}: {str(e)}")
        if not dncnn_loaded: print(f"DnCNN model not found. Searched in {args.model_load_path} for: {', '.join(dncnn_filenames)}")

        try:
            rednet_path = os.path.join(args.model_load_path, rednet_filename)
            if os.path.exists(rednet_path):
                rednet_model_eval.load_state_dict(torch.load(rednet_path, map_location=DEVICE))
                rednet_model_eval.to(DEVICE).eval()
                rednet_loaded = True
                print(f"REDNet model loaded from {rednet_path}")
        except Exception as e:
            print(f"Error loading {rednet_filename}: {str(e)}")
        if not rednet_loaded: print(f"REDNet model ({rednet_filename}) not found in {args.model_load_path}. Skipping REDNet evaluation.")
        
        for i in range(len(eval_fbps_gt)):
            current_fbp_gt = eval_fbps_gt[i]
            current_fbp_noisy_for_comparison = eval_fbps_noisy[i]
            current_full_noisy_sino_eval = eval_sinos_noisy_all[i]
            current_eval_angles = eval_angles_list[i]
            print(f"\n--- Evaluating on Phantom {i+1} ---")

            denoised_unet_n2i_output, denoised_dncnn_n2i_output, denoised_rednet_n2i_output = None, None, None
            residual_unet_img, residual_dncnn_img, residual_rednet_img = None, None, None

            eval_split_sinos = [current_full_noisy_sino_eval[:, s_idx::args.k_splits] for s_idx in range(args.k_splits)]
            eval_split_angles = [current_eval_angles[s_idx::args.k_splits] for s_idx in range(args.k_splits)]
            eval_input_sub_recons_raw = [reconstruct_fbp(s, a) for s, a in zip(eval_split_sinos, eval_split_angles)]
            eval_input_sub_recons = [
                resize(recon, (PHANTOM_SIZE_XY, PHANTOM_SIZE_XY), anti_aliasing=True, mode='reflect').astype(np.float32)
                for recon in eval_input_sub_recons_raw
            ]

            with torch.no_grad():
                if unet_loaded:
                    unet_outputs_to_average = []
                    for k_main_loop in range(args.k_splits):
                        current_input_indices = [j_idx for j_idx in range(args.k_splits) if j_idx != k_main_loop]
                        if not current_input_indices: continue
                        input_recons_to_avg_eval = [eval_input_sub_recons[j_idx] for j_idx in current_input_indices]
                        network_input_eval = np.mean(input_recons_to_avg_eval, axis=0)
                        input_tensor = torch.from_numpy(network_input_eval).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        unet_outputs_to_average.append(unet_model_eval(input_tensor).squeeze().cpu().numpy())
                    if unet_outputs_to_average:
                        denoised_unet_n2i_output = np.mean(unet_outputs_to_average, axis=0)
                        residual_unet_img = denoised_unet_n2i_output - current_fbp_gt
                        print("\nU-Net N2I Evaluation Results:")
                        evaluate_denoising_metrics(denoised_unet_n2i_output, current_fbp_gt, current_fbp_noisy_for_comparison)

                if dncnn_loaded:
                    dncnn_outputs_to_average = []
                    for k_main_loop in range(args.k_splits):
                        current_input_indices = [j_idx for j_idx in range(args.k_splits) if j_idx != k_main_loop]
                        if not current_input_indices: continue
                        input_recons_to_avg_eval = [eval_input_sub_recons[j_idx] for j_idx in current_input_indices]
                        network_input_eval = np.mean(input_recons_to_avg_eval, axis=0)
                        input_tensor = torch.from_numpy(network_input_eval).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        dncnn_outputs_to_average.append(dncnn_model_eval(input_tensor).squeeze().cpu().numpy())
                    if dncnn_outputs_to_average:
                        denoised_dncnn_n2i_output = np.mean(dncnn_outputs_to_average, axis=0)
                        residual_dncnn_img = denoised_dncnn_n2i_output - current_fbp_gt
                        print("\nDnCNN N2I Evaluation Results:")
                        evaluate_denoising_metrics(denoised_dncnn_n2i_output, current_fbp_gt, current_fbp_noisy_for_comparison)
                
                if rednet_loaded:
                    rednet_outputs_to_average = []
                    for k_main_loop in range(args.k_splits):
                        current_input_indices = [j_idx for j_idx in range(args.k_splits) if j_idx != k_main_loop]
                        if not current_input_indices: continue
                        input_recons_to_avg_eval = [eval_input_sub_recons[j_idx] for j_idx in current_input_indices]
                        network_input_eval = np.mean(input_recons_to_avg_eval, axis=0)
                        input_tensor = torch.from_numpy(network_input_eval).unsqueeze(0).unsqueeze(0).to(DEVICE)
                        rednet_outputs_to_average.append(rednet_model_eval(input_tensor).squeeze().cpu().numpy())
                    if rednet_outputs_to_average:
                        denoised_rednet_n2i_output = np.mean(rednet_outputs_to_average, axis=0)
                        residual_rednet_img = denoised_rednet_n2i_output - current_fbp_gt
                        print("\nREDNet N2I Evaluation Results:")
                        evaluate_denoising_metrics(denoised_rednet_n2i_output, current_fbp_gt, current_fbp_noisy_for_comparison)


            if args.plot_data:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                common_kwargs = {'cmap': 'gray', 'vmin': np.min(current_fbp_gt), 'vmax': np.max(current_fbp_gt)}
                
                axes[0, 0].imshow(current_fbp_gt, **common_kwargs)
                axes[0, 0].set_title("Ground Truth FBP (Clean)"); axes[0, 0].axis('off')
                axes[0, 1].imshow(current_fbp_noisy_for_comparison, **common_kwargs)
                axes[0, 1].set_title(f"Noisy FBP ({args.noise_type_arg})"); axes[0, 1].axis('off')
                
                res_plot_vmax = 0.3*np.max(current_fbp_gt)
                if residual_dncnn_img is not None:
                    axes[0, 2].imshow(residual_dncnn_img, cmap='coolwarm', vmin=-res_plot_vmax, vmax=res_plot_vmax)
                    axes[0, 2].set_title("Residual (DnCNN - GT)"); axes[0, 2].axis('off')
                else:
                    axes[0, 2].imshow(np.zeros_like(current_fbp_gt), cmap='coolwarm'); axes[0, 2].set_title("Residual (DnCNN - GT) (Missing)"); axes[0, 2].axis('off')
                
                if denoised_unet_n2i_output is not None:
                    axes[1, 0].imshow(denoised_unet_n2i_output, **common_kwargs)
                    axes[1, 0].set_title("Denoised by U-Net"); axes[1, 0].axis('off')
                else:
                    axes[1, 0].imshow(np.zeros_like(current_fbp_gt), **common_kwargs); axes[1, 0].set_title("Denoised by U-Net (Missing)"); axes[1, 0].axis('off')
                
                if residual_unet_img is not None:
                    axes[1, 1].imshow(residual_unet_img, cmap='coolwarm', vmin=-res_plot_vmax, vmax=res_plot_vmax)
                    axes[1, 1].set_title("Residual (U-Net - GT)"); axes[1, 1].axis('off')
                else:
                    axes[1, 1].imshow(np.zeros_like(current_fbp_gt), cmap='coolwarm'); axes[1, 1].set_title("Residual (U-Net - GT) (Missing)"); axes[1, 1].axis('off')
                
                if denoised_dncnn_n2i_output is not None:
                    axes[1, 2].imshow(denoised_dncnn_n2i_output, **common_kwargs)
                    axes[1, 2].set_title("Denoised by DnCNN"); axes[1, 2].axis('off')
                else:
                    axes[1, 2].imshow(np.zeros_like(current_fbp_gt), **common_kwargs); axes[1, 2].set_title("Denoised by DnCNN (Missing)"); axes[1, 2].axis('off')
                
                plt.tight_layout()
                plt.suptitle("Visual Comparison of Denoising Results (U-Net & DnCNN Focus)", fontsize=16)
                plt.subplots_adjust(top=0.92)
                plt.show()
                
                num_residuals_to_plot = sum(x is not None for x in [residual_unet_img, residual_dncnn_img, residual_rednet_img])
                if num_residuals_to_plot > 0:
                    plt.figure(figsize=(5 * num_residuals_to_plot, 4))
                    plot_idx = 1
                    res_vmax = 0.3 * np.max(current_fbp_gt)
                    res_vmin = -res_vmax
                    if residual_unet_img is not None:
                        plt.subplot(1, num_residuals_to_plot, plot_idx)
                        plt.imshow(residual_unet_img, cmap='coolwarm', vmin=res_vmin, vmax=res_vmax)
                        plt.title("Residual (U-Net - GT)"); plt.axis('off'); plt.colorbar()
                        plot_idx += 1
                    if residual_dncnn_img is not None:
                        plt.subplot(1, num_residuals_to_plot, plot_idx)
                        plt.imshow(residual_dncnn_img, cmap='coolwarm', vmin=res_vmin, vmax=res_vmax)
                        plt.title("Residual (DnCNN - GT)"); plt.axis('off'); plt.colorbar()
                        plot_idx += 1
                    if residual_rednet_img is not None:
                        plt.subplot(1, num_residuals_to_plot, plot_idx)
                        plt.imshow(residual_rednet_img, cmap='coolwarm', vmin=res_vmin, vmax=res_vmax)
                        plt.title("Residual (REDNet - GT)"); plt.axis('off'); plt.colorbar()
                    plt.tight_layout()
                    plt.suptitle("Comparative Residuals", fontsize=14)
                    plt.subplots_adjust(top=0.85 if num_residuals_to_plot > 1 else 0.75)
                    plt.show()
                
                if denoised_unet_n2i_output is not None:
                    fig_unet, ax_unet = plt.subplots(1, 3, figsize=(15, 5))
                    ax_unet[0].imshow(current_fbp_gt, **common_kwargs); ax_unet[0].set_title("Ground Truth FBP"); ax_unet[0].axis('off')
                    ax_unet[1].imshow(current_fbp_noisy_for_comparison, **common_kwargs); ax_unet[1].set_title("Noisy FBP"); ax_unet[1].axis('off')
                    ax_unet[2].imshow(denoised_unet_n2i_output, **common_kwargs); ax_unet[2].set_title("U-Net Denoised"); ax_unet[2].axis('off')
                    plt.tight_layout(); plt.suptitle("U-Net Denoising Results", fontsize=14); plt.subplots_adjust(top=0.85); plt.show()
                
                if denoised_dncnn_n2i_output is not None:
                    fig_dncnn, ax_dncnn = plt.subplots(1, 3, figsize=(15, 5))
                    ax_dncnn[0].imshow(current_fbp_gt, **common_kwargs); ax_dncnn[0].set_title("Ground Truth FBP"); ax_dncnn[0].axis('off')
                    ax_dncnn[1].imshow(current_fbp_noisy_for_comparison, **common_kwargs); ax_dncnn[1].set_title("Noisy FBP"); ax_dncnn[1].axis('off')
                    ax_dncnn[2].imshow(denoised_dncnn_n2i_output, **common_kwargs); ax_dncnn[2].set_title("DnCNN Denoised"); ax_dncnn[2].axis('off')
                    plt.tight_layout(); plt.suptitle("DnCNN Denoising Results", fontsize=14); plt.subplots_adjust(top=0.85); plt.show()

                if denoised_rednet_n2i_output is not None:
                    fig_rednet, ax_rednet = plt.subplots(1, 3, figsize=(15, 5))
                    ax_rednet[0].imshow(current_fbp_gt, **common_kwargs); ax_rednet[0].set_title("Ground Truth FBP"); ax_rednet[0].axis('off')
                    ax_rednet[1].imshow(current_fbp_noisy_for_comparison, **common_kwargs); ax_rednet[1].set_title("Noisy FBP"); ax_rednet[1].axis('off')
                    ax_rednet[2].imshow(denoised_rednet_n2i_output, **common_kwargs); ax_rednet[2].set_title("REDNet Denoised"); ax_rednet[2].axis('off')
                    plt.tight_layout(); plt.suptitle("REDNet Denoising Results", fontsize=14); plt.subplots_adjust(top=0.85); plt.show()


    print("\nScript finished.")
    if args.action == 'generate_data': print(f"Data generation complete. Files saved in {args.data_save_path}")
    print("If denoising is still suboptimal: try more epochs, more training phantoms, tune learning rate/batch size, or adjust K-splits for N2I.")