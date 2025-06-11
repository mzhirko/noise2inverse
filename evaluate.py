from utils.prepare_data import generate_phantom, prepare_training_data
import matplotlib.pyplot as plt
from utils.models import DnCNN, UNet
import os
import torch
import numpy as np
import argparse
import random
from PIL import Image
import glob

random.seed(42)
np.random.seed(42)
# Parse command line arguments
parser = argparse.ArgumentParser(description='Model comparison - Image Grid')
parser.add_argument('-p', '--phantom', action='store_true', 
                    help='Use phantom test image (default: use random CT dataset image)')
parser.add_argument('-m', '--model', choices=['DnCNN', 'UNet'], default='DnCNN',
                    help='Model architecture to use (default: DnCNN)')
args = parser.parse_args()

# Define the paths to your trained models
DnCNN_models_paths = [
    "/data/s4283341/phantom_DnCNN_256/dncnn_views_256_trained.pth",
    "/data/s4283341/phantom_DnCNN_512/dncnn_views_512_trained.pth",
    "/data/s4283341/phantom_DnCNN_1024/dncnn_views_1024_trained.pth"
]

UNet_models_paths = [
    "/data/s4283341/phantom_UNet_256/unet_views_256_trained.pth",
    "/data/s4283341/phantom_UNet_512/unet_views_512_trained.pth",
    "/data/s4283341/phantom_UNet_1024/unet_views_1024_trained.pth"
]

# Select model paths and initialize model based on argument
if args.model == 'DnCNN':
    models_paths = DnCNN_models_paths
    model = DnCNN(in_channels=1, out_channels=1)
    model_arch_name = "DnCNN"
elif args.model == 'UNet':
    models_paths = UNet_models_paths
    model = UNet(in_channels=1, out_channels=1)
    model_arch_name = "UNet"

# Define the corresponding number of views for each model
model_views = [256, 512, 1024]

# Generate or load the test image
if args.phantom:
    # Use the generated phantom
    test_image = generate_phantom()
    image_source = "Generated Phantom"
else:
    # Randomly pick an image from the CT dataset
    ct_dataset_path = "/local/s4283341/cito/CT_dataset"
    
    # Find all image files (assuming common formats)
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(ct_dataset_path, '**', pattern), recursive=True))
    
    if not all_images:
        raise ValueError(f"No images found in {ct_dataset_path}")
    
    # Randomly select an image
    selected_image_path = random.choice(all_images)
    print(f"Selected image: {selected_image_path}")
    
    # Load and resize the image to 256x256
    pil_image = Image.open(selected_image_path)
    
    # Convert to grayscale if needed
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Resize to 256x256
    pil_image = pil_image.resize((256, 256), Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    test_image = np.array(pil_image).astype(np.float32) / 255.0
    
    image_source = f"CT Dataset: {os.path.basename(selected_image_path)}"

# Create a 3x3 grid: 3 models (rows) x 3 columns (Noisy Input | Reconstruction | Ground Truth)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Column headers
col_titles = ["Noisy Input", "Reconstruction", "Ground Truth"]

for i, (model_path, n_views) in enumerate(zip(models_paths, model_views)):
    # Load the state dictionary of the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    model_name = os.path.basename(os.path.dirname(model_path))  # Gets the parent directory name
    
    # Prepare data for the specific number of views corresponding to this model
    input_img_np, _ = prepare_training_data(test_image, k=4, n_views=n_views, noise_level=0.05)

    # Convert numpy input to tensor and add batch dimension (N, C, H, W)
    input_img = torch.from_numpy(input_img_np).float().unsqueeze(0).unsqueeze(0)

    # --- Perform Inference ---
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(input_img)

    # Transfer output back to numpy and remove batch and channel dimensions if present
    output_np = output.squeeze().cpu().numpy()

    # Column 0: Noisy Input
    axes[i, 0].imshow(input_img_np, cmap='gray')
    axes[i, 0].axis('off')
    
    # Column 1: Reconstruction
    axes[i, 1].imshow(output_np, cmap='gray')
    axes[i, 1].axis('off')
    
    # Column 2: Ground Truth
    axes[i, 2].imshow(test_image, cmap='gray')
    axes[i, 2].axis('off')
    
    # Set column titles for top row
    if i == 0:
        for j, title in enumerate(col_titles):
            axes[i, j].set_title(title, fontsize=12)
    
    # Set row labels for left column (model info with views)
    if True:  # Always show for all rows
        axes[i, 0].text(-0.15, 0.5, f"{model_arch_name}\n({n_views} views)", 
                        transform=axes[i, 0].transAxes,
                        fontsize=10, va='center', ha='right', rotation=0)

# plt.suptitle(f"{model_arch_name} Model Comparison - {image_source}", fontsize=16, y=0.95)
plt.tight_layout()

# Create output filename with image type prefix
image_type = "phantom" if args.phantom else "CT"
output_filename = f'{image_type}_{model_arch_name.lower()}_comparison_grid.png'

plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

print("Image saved:")
print(f"- {output_filename}")