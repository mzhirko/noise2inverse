from utils.phantom import generate_phantom
import numpy as np
import random
from skimage.draw import ellipse, rectangle, polygon, disk
import matplotlib.pyplot as plt
import astra
import torch
from collections import defaultdict
import os
from torch.utils.data import Dataset
from skimage.transform import resize

def create_sinogram(phantom, n_views):
    n_rows, n_cols = phantom.shape
    vol_geom = astra.create_vol_geom(n_cols, n_rows)
    
    detector_count = int(np.ceil(max(phantom.shape[0], phantom.shape[1]) * np.sqrt(2)))
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detector_count, angles)
    
    projector_type_str = "cuda" if torch.cuda.is_available() else "linear"
    projector_id = astra.create_projector(projector_type_str, proj_geom, vol_geom)
    
    sinogram_id, sinogram_data = astra.create_sino(phantom, projector_id)
    return sinogram_id, sinogram_data, projector_id, vol_geom

def reconstruct_from_sino(sinogram_data_full, k, full_projector_id, vol_geom):
    origin_proj_geom_dict = astra.projector.projection_geometry(full_projector_id)
    detector_count = origin_proj_geom_dict["DetectorCount"]
    all_angles_full = origin_proj_geom_dict['ProjectionAngles']
    detector_spacing = origin_proj_geom_dict.get('DetectorSpacingX', 1.0) 
    
    projector_engine_type = "cuda" if torch.cuda.is_available() else "linear"
    fbp_algo_type = 'FBP_CUDA' if projector_engine_type == 'cuda' else 'FBP'

    grouped_sino_data_slices = defaultdict(list)
    grouped_angles_list = defaultdict(list)
    
    for i in range(sinogram_data_full.shape[0]):
        group_number = i % k
        grouped_sino_data_slices[group_number].append(sinogram_data_full[i])
        grouped_angles_list[group_number].append(all_angles_full[i])
        
    group_projector_ids = {}
    group_sino_astra_ids = {}
    
    for group_idx in grouped_sino_data_slices.keys():
        current_sino_slices_np = np.stack(grouped_sino_data_slices[group_idx])
        current_angles_np = np.array(grouped_angles_list[group_idx])

        if current_angles_np.size == 0:
            continue

        group_proj_geom = astra.create_proj_geom('parallel', detector_spacing, detector_count, current_angles_np)
        
        p_id = astra.create_projector(projector_engine_type, group_proj_geom, vol_geom)
        group_projector_ids[group_idx] = p_id
        
        s_id = astra.data2d.create('-sino', group_proj_geom, current_sino_slices_np)
        group_sino_astra_ids[group_idx] = s_id
            
    reconstructions_list = []
    for group_idx in grouped_sino_data_slices.keys(): 
        if group_idx not in group_sino_astra_ids: 
            reconstructions_list.append(None) 
            continue

        recon_vol_id = astra.data2d.create('-vol', vol_geom, 0)
        
        cfg = astra.astra_dict(fbp_algo_type)
        cfg['ProjectorId'] = group_projector_ids[group_idx]
        cfg['ProjectionDataId'] = group_sino_astra_ids[group_idx]
        cfg['ReconstructionDataId'] = recon_vol_id
        
        fbp_alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fbp_alg_id)
        reconstructions_list.append(astra.data2d.get(recon_vol_id))
        
        astra.data2d.delete(recon_vol_id)
        astra.algorithm.delete(fbp_alg_id)

    for group_idx in group_projector_ids.keys():
        if group_idx in group_projector_ids: 
             astra.projector.delete(group_projector_ids[group_idx])
    for group_idx in group_sino_astra_ids.keys():
        if group_idx in group_sino_astra_ids: 
            astra.data2d.delete(group_sino_astra_ids[group_idx])
            
    return reconstructions_list

def prepare_training_data(phantom, k=2, n_views=360, noise_level=0.01):
    sinogram_id_main, sinogram_data_main, projector_id_main, vol_geom_main = create_sinogram(phantom, n_views)
    noise = np.random.normal(0, noise_level * np.max(sinogram_data_main), sinogram_data_main.shape)
    noisy_sinogram_data_main = sinogram_data_main + noise
    all_recs = reconstruct_from_sino(noisy_sinogram_data_main, k, projector_id_main, vol_geom_main)
    
    target_idx = random.randrange(len(all_recs))
    target_img = all_recs[target_idx]
    input_img = np.mean(np.stack([rec for i, rec in enumerate(all_recs) if i != target_idx]), axis=0)
    return input_img, target_img

def create_phantom_dataset(n, n_views, noise_level, k_splits=2):
    dataset = []
    for _ in range(n):
        phantom_img = generate_phantom() # Assumes generate_phantom() creates a 2D numpy array
        input_img, target_img = prepare_training_data(phantom_img, k=k_splits, n_views=n_views, noise_level=noise_level)
        if input_img is not None and target_img is not None:
            dataset.append({'input': input_img, 'target': target_img, 'original_image': phantom_img})
    return dataset

def create_images_dataset(image_folder_path, target_size_n, n_views, noise_level, k_splits=2):
    dataset = []
    if not os.path.isdir(image_folder_path):
        print(f"Error: Provided path '{image_folder_path}' is not a valid directory.")
        return dataset

    for filename in os.listdir(image_folder_path):
        file_path = os.path.join(image_folder_path, filename)
        try:
            img_data = plt.imread(file_path)

            # Grayscale conversion
            if img_data.ndim == 3:
                if img_data.shape[-1] == 4:  
                    img_data = img_data[..., :3]  
                if img_data.shape[-1] == 3:  
                    img_data = np.mean(img_data, axis=2)
            elif img_data.ndim == 2:
                pass 
            
            img_resized = resize(img_data, 
                                    (target_size_n, target_size_n), 
                                    anti_aliasing=True)

            img_processed = img_resized.astype(np.float32)
            
            if np.max(img_processed) > 1.0 + 1e-6:
                img_processed = img_processed / 255.0
                
            img_final = np.clip(img_processed, 0.0, 1.0)
            
            input_img, target_img = prepare_training_data(img_final, k=k_splits, n_views=n_views, noise_level=noise_level)
            if input_img is not None and target_img is not None:
                dataset.append({'input': input_img, 'target': target_img, 'original_image': img_final})
        except Exception as e:
            print(f"Skipping file {filename} due to an error during processing: {e}")
            
    return dataset

class ReconstructionDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        input_image = sample['input']
        target_image = sample['target']
        original_data = sample['original_image']

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
            original_data = self.transform(original_data)
            
        if input_image.ndim == 2:
            input_image = np.expand_dims(input_image, axis=0)
        if target_image.ndim == 2:
            target_image = np.expand_dims(target_image, axis=0)
        if original_data.ndim == 2:
            original_data = np.expand_dims(original_data, axis=0)

        return torch.from_numpy(input_image.copy()), torch.from_numpy(target_image.copy()), torch.from_numpy(original_data.copy())

