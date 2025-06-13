from utils.prepare_data import generate_phantom, prepare_training_data
import matplotlib.pyplot as plt
from utils.models import DnCNN
import os
import torch
import numpy as np
import argparse
import random
from PIL import Image
import glob

# This script evaluates how DnCNN models trained on a specific number of views
# perform when tested on data generated with different numbers of views.
# For example, it checks how a model trained on 256 views reconstructs an image
# from 256, 512, and 1024 views.

parser = argparse.ArgumentParser(description='DnCNN model comparison')
parser.add_argument('-p', '--phantom', action='store_true', 
                    help='Use phantom test image (default: use random CT dataset image)')
args = parser.parse_args()

# Initialize the DnCNN model.
model = DnCNN(in_channels=1, out_channels=1)

# --- Model Paths ---
# Paths to models trained on phantoms.
phantom_models_paths = [
    "./trained_models/views_256/DnCNN/dncnn_views_256_trained.pth",
    "./trained_models/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models/views_1024/DnCNN/dncnn_views_1024_trained.pth",
]
# Paths to models trained on real CT images.
CT_models_paths = [
    "./trained_models_images/views_256/DnCNN/dncnn_views_256_trained.pth",
    "./trained_models_images/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models_images/views_1024/DnCNN/dncnn_views_1024_trained.pth",
]
# Select paths based on the '--phantom' argument.
models_paths = phantom_models_paths if args.phantom else CT_models_paths

# --- Test Image Preparation ---
if args.phantom:
    test_image = generate_phantom()
    image_source = "Generated Phantom"
else:
    ct_dataset_path = "./CT_dataset"
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.bmp']
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(os.path.join(ct_dataset_path, '**', pattern), recursive=True))
    if not all_images:
        raise ValueError(f"No images found in {ct_dataset_path}")
    selected_image_path = random.choice(all_images)
    print(f"Selected image: {selected_image_path}")
    
    pil_image = Image.open(selected_image_path).convert('L')
    pil_image = pil_image.resize((256, 256), Image.LANCZOS)
    test_image = np.array(pil_image).astype(np.float32) / 255.0
    image_source = f"CT Dataset: {os.path.basename(selected_image_path)}"

# The different numbers of views to use for generating the test data.
n_views_list = [256, 512, 1024]

# --- Generate and Save Comparison Grid ---

# First, save the ground truth image separately for reference.
plt.figure(figsize=(6, 6))
plt.imshow(test_image, cmap='gray')
plt.title(f"Ground Truth: {image_source}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{"phantom" if args.phantom else "CT"}_ground_truth.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a 3x3 grid for the cross-view comparison.
# Rows: Models trained on 256, 512, 1024 views.
# Columns: Test data generated with 256, 512, 1024 views.
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Iterate over each pre-trained model (rows).
for i, model_path in enumerate(models_paths):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Extract model name for labeling, e.g., 'dncnn_views_256_trained'
    model_name = os.path.basename(model_path).split('.')[0]
    
    # Iterate over each number of views for the input data (columns).
    for j, n_views in enumerate(n_views_list):
        # Prepare the noisy input image using the current number of views.
        input_img_np, _ = prepare_training_data(test_image, k=4, n_views=n_views, noise_level=0.05)
        input_img = torch.from_numpy(input_img_np).float().unsqueeze(0).unsqueeze(0)

        # Perform inference with the loaded model.
        with torch.no_grad():
            output = model(input_img)
        output_np = output.squeeze().cpu().numpy()

        # Display the reconstructed image in the grid.
        axes[i, j].imshow(output_np, cmap='gray')
        axes[i, j].axis('off')
        
        # Set column titles (Input Views) for the top row.
        if i == 0:
            axes[i, j].set_title(f"Input: {n_views} Views", fontsize=12)
        
        # Set row labels (Model Trained On) for the first column.
        if j == 0:
            trained_on_views = models_paths[i].split('views_')[1].split('/')[0]
            axes[i, j].text(-0.2, 0.5, f"Model Trained\non {trained_on_views} Views", 
                            transform=axes[i, j].transAxes,
                            fontsize=10, va='center', ha='right', rotation=0)

plt.tight_layout()
plt.savefig(f'{"phantom" if args.phantom else "CT"}_dncnn_cross_view_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Images saved:")
print(f'- {"phantom" if args.phantom else "CT"}_ground_truth.png')
print(f'- {"phantom" if args.phantom else "CT"}_dncnn_cross_view_comparison.png')