import numpy as np
import random
from skimage.draw import ellipse, rectangle, polygon, disk
import matplotlib.pyplot as plt

PHANTOM_SIZE_XY = 256

def _can_place_disk(r_candidate, c_candidate, rad_candidate, size, placed_disks):
    """Helper function to check if a new disk overlaps with existing ones or image boundaries."""
    # Check if the disk is fully within the image boundaries.
    if not (r_candidate - rad_candidate >= 0 and \
            r_candidate + rad_candidate < size and \
            c_candidate - rad_candidate >= 0 and \
            c_candidate + rad_candidate < size):
        return False
    # Check for overlap with any previously placed disks.
    for disk_info in placed_disks:
        dist_sq = (r_candidate - disk_info['r'])**2 + (c_candidate - disk_info['c'])**2
        sum_rad = rad_candidate + disk_info['rad']
        if dist_sq < sum_rad * sum_rad:
            return False
    return True

def generate_phantom(size=PHANTOM_SIZE_XY):
    """Generates a complex, randomized phantom image with various shapes and textures."""
    phantom = np.zeros((size, size), dtype=np.float32)
    all_placed_disks = []
    MAX_PLACEMENT_ATTEMPTS_PER_DISK = 20 

    # --- Define random parameters for this specific phantom ---
    num_regions_actual = random.randint(2, 4)
    details_per_region_actual = random.randint(8, 15) 
    foam_elements_actual = random.randint(30, 60)

    # --- Generate large background regions ---
    for _ in range(num_regions_actual):
        region_type = random.choice(['mixed_shapes', 'foam_area'])
        # Define a random rectangular area for the region.
        r_start, c_start = random.randint(0, size // 2), random.randint(0, size // 2)
        r_end, c_end = random.randint(r_start + size // 4, size - 1), random.randint(c_start + size // 4, size - 1)
        region_value_base = random.uniform(0.1, 0.7)

        # Region with various geometric shapes.
        if region_type == 'mixed_shapes':
            for _ in range(details_per_region_actual):
                shape_type = random.choice(['ellipse', 'rectangle', 'polygon'])
                value = region_value_base + random.uniform(-0.1, 0.5)
                r_c, c_c = random.randint(r_start, r_end), random.randint(c_start, c_end)

                if shape_type == 'ellipse':
                    rr, cc = ellipse(r_c, c_c, random.randint(max(1,size // 32), size // 10), 
                                     random.randint(max(1,size // 32), size // 10), 
                                     shape=(size, size), rotation=random.uniform(0, np.pi))
                    phantom[rr, cc] = value
                elif shape_type == 'rectangle':
                    width, height = random.randint(size // 20, size // 8), random.randint(size // 20, size // 8)
                    s_r, s_c = max(0, r_c - height//2), max(0, c_c - width//2)
                    e_r, e_c = min(size-1, s_r + height), min(size-1, s_c + width)
                    if e_r > s_r and e_c > s_c:
                        rr, cc = rectangle((s_r, s_c), end=(e_r, e_c), shape=(size, size))
                        phantom[rr, cc] = value
                elif shape_type == 'polygon':
                    num_v = random.randint(3,5)
                    verts_r = np.clip(r_c + np.random.randint(-size//10, size//10, num_v), 0, size-1)
                    verts_c = np.clip(c_c + np.random.randint(-size//10, size//10, num_v), 0, size-1)
                    if len(verts_r) >= 3:
                        rr, cc = polygon(verts_r, verts_c, shape=(size,size))
                        phantom[rr,cc] = value
        
        # Region with a foam-like texture (overlapping disks).
        elif region_type == 'foam_area':
            rr_foam_bg, cc_foam_bg = rectangle((r_start, c_start), end=(r_end, c_end), shape=(size,size))
            phantom[rr_foam_bg, cc_foam_bg] = np.maximum(phantom[rr_foam_bg, cc_foam_bg], region_value_base * 0.5)

            for _ in range(foam_elements_actual):
                val_foam = region_value_base + random.uniform(-0.25, 0.25)
                for _ in range(MAX_PLACEMENT_ATTEMPTS_PER_DISK):
                    r_f, c_f = random.randint(r_start, r_end), random.randint(c_start, c_end)
                    rad_f = random.randint(max(1,size // 64), size // 20)
                    if _can_place_disk(r_f, c_f, rad_f, size, all_placed_disks):
                        rr, cc = disk((r_f, c_f), rad_f, shape=(size,size))
                        phantom[rr,cc] = np.clip(val_foam, 0, 1.5)
                        all_placed_disks.append({'r': r_f, 'c': c_f, 'rad': rad_f})
                        break

    # --- Add small, high-contrast details on top of the existing structures ---
    for _ in range(details_per_region_actual * 2):
        val_detail = random.uniform(1.0, 1.5)
        for _ in range(MAX_PLACEMENT_ATTEMPTS_PER_DISK):
            r_d, c_d = random.randint(0, size-1), random.randint(0, size-1)
            rad_d = random.randint(1, max(2, size // 40))
            if _can_place_disk(r_d, c_d, rad_d, size, all_placed_disks):
                rr, cc = disk((r_d, c_d), rad_d, shape=(size,size))
                phantom[rr,cc] = val_detail
                all_placed_disks.append({'r': r_d, 'c': c_d, 'rad': rad_d})
                break
                
    return np.clip(phantom, 0, 1.5)