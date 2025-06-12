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

# Parse command line arguments
parser = argparse.ArgumentParser(description='DnCNN model comparison')
parser.add_argument('-p', '--phantom', action='store_true', 
                    help='Use phantom test image (default: use random CT dataset image)')
args = parser.parse_args()

# Initialize the DnCNN model (assuming in_channels and out_channels are 1 for grayscale images)
model = DnCNN(in_channels=1, out_channels=1)

# Define the paths to your trained models
phantom_models_paths = [
    "./trained_models/views_1024/DnCNN/dncnn_views_1024_trained.pth",
    "./trained_models/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models/views_256/DnCNN/dncnn_views_256_trained.pth"
]

CT_models_paths = [
    "./trained_models_images/views_1024/DnCNN/dncnn_views_1024_trained.pth",
    "./trained_models_images/views_512/DnCNN/dncnn_views_512_trained.pth",
    "./trained_models_images/views_256/DnCNN/dncnn_views_256_trained.pth"
]

models_paths = phantom_models_paths if args.phantom else CT_models_paths
# Generate or load the test image
if args.phantom:
    # Use the generated phantom
    test_image = generate_phantom()
    image_source = "Generated Phantom"
else:
    # Randomly pick an image from the CT dataset
    ct_dataset_path = "./CT_dataset"
    
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

# Define the number of views to test
n_views_list = [256, 512, 1024]

# Create a 3x3 grid: 3 models (rows) x 3 views (columns)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# First, save the ground truth separately
plt.figure(figsize=(6, 6))
plt.imshow(test_image, cmap='gray')
plt.title(f"Ground Truth: {image_source}")
plt.axis('off')
plt.tight_layout()
plt.savefig(f'{"phantom" if args.phantom else "CT"}_ground_truth.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Now create the 3x3 grid for model outputs
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i, model_path in enumerate(models_paths):
    # Load the state dictionary of the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    model_name = os.path.basename(os.path.dirname(model_path))  # Gets the parent directory name like 'phantom_DnCNN_256'
    
    for j, n_views in enumerate(n_views_list):
        # Prepare data for a specific number of views and noise level
        # input_img_np will be the noisy image, target_img_np is the clean ground truth (test_image)
        input_img_np, _ = prepare_training_data(test_image, k=4, n_views=n_views, noise_level=0.05)

        # Convert numpy input to tensor and add batch dimension (N, C, H, W)
        input_img = torch.from_numpy(input_img_np).float().unsqueeze(0).unsqueeze(0)

        # --- Perform Inference ---
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model(input_img)

        # Transfer output back to numpy and remove batch and channel dimensions if present
        output_np = output.squeeze().cpu().numpy()

        # --- Display the Model's Denoised Output ---
        axes[i, j].imshow(output_np, cmap='gray')
        axes[i, j].axis('off')
        
        # Set title for top row (different views)
        if i == 0:
            axes[i, j].set_title(f"{n_views} Views", fontsize=12)
        
        # Set row labels for left column (different models)
        if j == 0:
            axes[i, j].text(-0.1, 0.5, model_name, transform=axes[i, j].transAxes,
                            fontsize=10, va='center', ha='right', rotation=90)

plt.tight_layout()
plt.savefig(f'{"phantom" if args.phantom else "CT"}_dncnn_comparison_3x3.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

print("Images saved:")
print(f'- {"phantom" if args.phantom else "CT"}_ground_truth.png')
print(f'- {"phantom" if args.phantom else "CT"}_dncnn_comparison_3x3.png')