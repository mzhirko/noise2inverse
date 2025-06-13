from utils.phantom import generate_phantom
import numpy as np
import random
import astra
import torch
from collections import defaultdict
import os
from torch.utils.data import Dataset
from skimage.transform import resize
import matplotlib.pyplot as plt

def create_sinogram(phantom, n_views):
    """Generates a sinogram from a phantom image using the ASTRA toolbox."""
    vol_geom = astra.create_vol_geom(phantom.shape[1], phantom.shape[0])
    detector_count = int(np.ceil(np.sqrt(2) * max(phantom.shape)))
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_count, angles)
    
    projector_type = "cuda" if torch.cuda.is_available() else "linear"
    projector_id = astra.create_projector(projector_type, proj_geom, vol_geom)
    
    sinogram_id, sinogram_data = astra.create_sino(phantom, projector_id)
    return sinogram_id, sinogram_data, projector_id, vol_geom

def reconstruct_from_sino(sinogram_data, k, projector_id_full, vol_geom):
    """Reconstructs images from a sinogram that is split into k subsets."""
    proj_geom_full = astra.projector.projection_geometry(projector_id_full)
    angles_full = proj_geom_full['ProjectionAngles']
    
    # Group sinogram slices and corresponding angles into k subsets.
    sino_groups = defaultdict(list)
    angle_groups = defaultdict(list)
    for i in range(sinogram_data.shape[0]):
        group_idx = i % k
        sino_groups[group_idx].append(sinogram_data[i])
        angle_groups[group_idx].append(angles_full[i])
        
    reconstructions = []
    fbp_algo = 'FBP_CUDA' if torch.cuda.is_available() else 'FBP'

    # Reconstruct an image for each subset of views.
    for group_idx in range(k):
        if not angle_groups[group_idx]: continue
        
        # Create a new geometry and projector for the subset of angles.
        group_angles = np.array(angle_groups[group_idx])
        group_proj_geom = astra.create_proj_geom('parallel', 1.0, proj_geom_full['DetectorCount'], group_angles)
        group_projector_id = astra.create_projector(proj_geom_full['type'], group_proj_geom, vol_geom)
        
        # Perform FBP reconstruction.
        group_sino_data = np.stack(sino_groups[group_idx])
        sino_id = astra.data2d.create('-sino', group_proj_geom, group_sino_data)
        recon_id = astra.data2d.create('-vol', vol_geom, 0)
        
        cfg = astra.astra_dict(fbp_algo)
        cfg.update({'ProjectorId': group_projector_id, 'ProjectionDataId': sino_id, 'ReconstructionDataId': recon_id})
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        
        reconstructions.append(astra.data2d.get(recon_id))
        
        # Clean up ASTRA objects.
        astra.data2d.delete([sino_id, recon_id])
        astra.projector.delete(group_projector_id)
        astra.algorithm.delete(alg_id)
            
    return reconstructions

def prepare_training_data(phantom, k, n_views, noise_level):
    """
    Creates a pair of (input, target) images for training.
    The process simulates sparse-view CT by splitting views into k sets. The input
    is the average of k-1 reconstructions, and the target is the k-th reconstruction.
    """
    sino_id, sino_data, proj_id, vol_geom = create_sinogram(phantom, n_views)
    noise = np.random.normal(0, noise_level * np.max(sino_data), sino_data.shape)
    noisy_sino_data = sino_data + noise
    all_recs = reconstruct_from_sino(noisy_sino_data, k, proj_id, vol_geom)
    
    astra.data2d.delete(sino_id)
    astra.projector.delete(proj_id)

    if not all_recs or len(all_recs) < k: return None, None

    # Randomly select one reconstruction as the target, and average the rest for the input.
    target_idx = random.randrange(len(all_recs))
    target_img = all_recs[target_idx]
    input_img = np.mean(np.stack([rec for i, rec in enumerate(all_recs) if i != target_idx]), axis=0)
    return input_img, target_img

def create_phantom_dataset(n, n_views, noise_level, k_splits):
    """Creates a dataset of size n using generated phantoms."""
    dataset = []
    for _ in range(n):
        phantom_img = generate_phantom()
        input_img, target_img = prepare_training_data(phantom_img, k=k_splits, n_views=n_views, noise_level=noise_level)
        if input_img is not None and target_img is not None:
            dataset.append({'input': input_img, 'target': target_img, 'original_image': phantom_img})
    return dataset

def create_images_dataset(image_folder_path, target_size_n, n_views, noise_level, k_splits):
    """Creates a dataset from a folder of real images."""
    dataset = []
    if not os.path.isdir(image_folder_path): return dataset

    for filename in os.listdir(image_folder_path):
        try:
            img_data = plt.imread(os.path.join(image_folder_path, filename))
            # Pre-process image: convert to grayscale, resize, and normalize to [0, 1].
            if img_data.ndim == 3:
                img_data = np.mean(img_data[..., :3], axis=2)
            img_resized = resize(img_data, (target_size_n, target_size_n), anti_aliasing=True)
            if np.max(img_resized) > 1.0: img_resized /= 255.0
            img_final = np.clip(img_resized, 0.0, 1.0).astype(np.float32)
            
            # Generate training pair from the processed image.
            input_img, target_img = prepare_training_data(img_final, k=k_splits, n_views=n_views, noise_level=noise_level)
            if input_img is not None and target_img is not None:
                dataset.append({'input': input_img, 'target': target_img, 'original_image': img_final})
        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")
    return dataset

class ReconstructionDataset(Dataset):
    """Custom PyTorch Dataset for the reconstruction task."""
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # Ensure images have a channel dimension and return as tensors.
        input_image = np.expand_dims(sample['input'], axis=0)
        target_image = np.expand_dims(sample['target'], axis=0)
        original_data = np.expand_dims(sample['original_image'], axis=0)
        return torch.from_numpy(input_image.copy()), torch.from_numpy(target_image.copy()), torch.from_numpy(original_data.copy())