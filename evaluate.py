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

# Set seeds for reproducibility.
random.seed(42)
np.random.seed(42)

# --- Argument Parsing ---
# This script generates a visual comparison of a model's performance.
parser = argparse.ArgumentParser(description='Model comparison - Image Grid')
# Argument to switch between using a generated phantom or a real CT image.
parser.add_argument('-p', '--phantom', action='store_true', 
                    help='Use phantom test image (default: use random CT dataset image)')
# Argument to select the model architecture (DnCNN or UNet).
parser.add_argument('-m', '--model', choices=['DnCNN', 'UNet'], default='DnCNN',
                    help='Model architecture to use (default: DnCNN)')
args = parser.parse_args()

# --- Model Paths ---
# Define paths to pre-trained models for different architectures and datasets.
UNet_phantom_models_paths = [
    "./trained_models/views_1024/UNet/unet_views_1024_trained.pth",
    "./trained_models/views_512/UNet/unet_views_512_trained.pth",
    "./trained_models/views_256/UNet/unet_views_256_trained.pth"
]

UNet_CT_models_paths = [
    "./trained_models_images/views_1024/UNet/unet_views_1024_trained.pth",
    "./trained_models_images/views_512/UNet/unet_views_512_trained.pth",
    "./trained_models_images/views_256/UNet/unet_views_256_trained.pth"
]

DnCNN_phantom_models_paths = [
    "./trained_models/views_1024/DnCNN/dncnn_views_1024_trained.pth",
    "./trained_models/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models/views_256/DnCNN/dncnn_views_256_trained.pth"
]

DnCNN_CT_models_paths = [
    "./trained_models_images/views_1024/DnCNN/dncnn_views_1024_trained.pth",
    "./trained_models_images/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models_images/views_256/DnCNN/dncnn_views_256_trained.pth"
]

# Select the appropriate model paths and initialize the model based on command-line arguments.
if args.model == 'DnCNN':
    models_paths = DnCNN_phantom_models_paths if args.phantom else DnCNN_CT_models_paths
    model = DnCNN(in_channels=1, out_channels=1)
    model_arch_name = "DnCNN"
elif args.model == 'UNet':
    models_paths = UNet_phantom_models_paths if args.phantom else UNet_CT_models_paths
    model = UNet(in_channels=1, out_channels=1)
    model_arch_name = "UNet"

# The number of views corresponding to each model. The order matches the model paths.
model_views = [1024, 512, 256]

# --- Test Image Preparation ---
# Generate or load the test image based on the '--phantom' argument.
if args.phantom:
    test_image = generate_phantom()
    image_source = "Generated Phantom"
else:
    ct_dataset_path = "./CT_dataset"
    # Find all image files in the dataset directory.
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(ct_dataset_path, '**', pattern), recursive=True))
    
    if not all_images:
        raise ValueError(f"No images found in {ct_dataset_path}")
    
    selected_image_path = random.choice(all_images)
    print(f"Selected image: {selected_image_path}")
    
    pil_image = Image.open(selected_image_path)
    
    # Ensure the image is grayscale.
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    
    # Resize to a standard size and normalize pixel values to [0, 1].
    pil_image = pil_image.resize((256, 256), Image.LANCZOS)
    test_image = np.array(pil_image).astype(np.float32) / 255.0
    image_source = f"CT Dataset: {os.path.basename(selected_image_path)}"

# --- Generate and Save Comparison Grid ---
# Create a 3x3 subplot grid. Rows for models, columns for image types.
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
col_titles = ["Noisy Input", "Reconstruction", "Ground Truth"]

# Iterate through each model to generate its reconstruction.
for i, (model_path, n_views) in enumerate(zip(models_paths, model_views)):
    # Load the pre-trained weights into the model.
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Prepare the noisy input data for the current number of views.
    input_img_np, _ = prepare_training_data(test_image, k=4, n_views=n_views, noise_level=0.05)
    input_img = torch.from_numpy(input_img_np).float().unsqueeze(0).unsqueeze(0)

    # Perform inference.
    with torch.no_grad():
        output = model(input_img)

    output_np = output.squeeze().cpu().numpy()

    # Populate the grid: Noisy Input, Reconstruction, Ground Truth.
    axes[i, 0].imshow(input_img_np, cmap='gray')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(output_np, cmap='gray')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(test_image, cmap='gray')
    axes[i, 2].axis('off')
    
    # Set titles for the columns on the first row.
    if i == 0:
        for j, title in enumerate(col_titles):
            axes[i, j].set_title(title, fontsize=12)
    
    # Set labels for each row, indicating the model and view count.
    axes[i, 0].text(-0.15, 0.5, f"{model_arch_name}\n({n_views} views)", 
                    transform=axes[i, 0].transAxes,
                    fontsize=10, va='center', ha='right', rotation=0)

plt.tight_layout()

# Save the generated figure to a file.
image_type = "phantom" if args.phantom else "CT"
output_filename = f'{image_type}_{model_arch_name.lower()}_comparison_grid.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()

print("Image saved:")
print(f"- {output_filename}")