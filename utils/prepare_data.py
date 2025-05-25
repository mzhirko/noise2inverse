from phantom import generate_phantom
import numpy as np
import random
from skimage.draw import ellipse, rectangle, polygon, disk
import matplotlib.pyplot as plt
import astra
import torch
from collections import defaultdict
import os

pt = generate_phantom()

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
