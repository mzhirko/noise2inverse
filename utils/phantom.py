import numpy as np
import random
from skimage.draw import ellipse, rectangle, polygon, disk
import matplotlib.pyplot as plt

PHANTOM_SIZE_XY = 512

def _can_place_disk(r_candidate, c_candidate, rad_candidate, size, placed_disks):
    if not (r_candidate - rad_candidate >= 0 and \
            r_candidate + rad_candidate < size and \
            c_candidate - rad_candidate >= 0 and \
            c_candidate + rad_candidate < size):
        return False

    for disk_info in placed_disks:
        dist_sq = (r_candidate - disk_info['r'])**2 + (c_candidate - disk_info['c'])**2
        sum_rad = rad_candidate + disk_info['rad']
        if dist_sq < sum_rad * sum_rad: # Overlap if distance squared is less than sum of radii squared
            return False
    return True

def generate_phantom(size=PHANTOM_SIZE_XY):
    phantom = np.zeros((size, size), dtype=np.float32)
    all_placed_disks = []
    MAX_PLACEMENT_ATTEMPTS_PER_DISK = 20 

    num_regions_actual = random.randint(2, 4)
    details_per_region_actual = random.randint(8, 15) 
    foam_elements_actual = random.randint(30, 60)

    for _ in range(num_regions_actual):
        region_type = random.choice(['mixed_shapes', 'foam_area'])
        r_start = random.randint(0, size // 2)
        c_start = random.randint(0, size // 2)
        r_end = random.randint(r_start + size // 4, size - 1)
        c_end = random.randint(c_start + size // 4, size - 1)

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
                val_foam = region_value_base + random.uniform(-0.25, 0.25)
                for attempt in range(MAX_PLACEMENT_ATTEMPTS_PER_DISK):
                    r_foam_center = random.randint(r_start, r_end)
                    c_foam_center = random.randint(c_start, c_end)
                    rad_foam = random.randint(max(1,size // 64), size // 20)

                    if _can_place_disk(r_foam_center, c_foam_center, rad_foam, size, all_placed_disks):
                        rr, cc = disk((r_foam_center, c_foam_center), rad_foam, shape=(size,size))
                        phantom[rr,cc] = np.clip(val_foam, 0, 1.5)
                        all_placed_disks.append({'r': r_foam_center, 'c': c_foam_center, 'rad': rad_foam})
                        break


    for _ in range(details_per_region_actual * 2):
        val_detail = random.uniform(1.0, 1.5)
        for attempt in range(MAX_PLACEMENT_ATTEMPTS_PER_DISK):
            r_detail_center = random.randint(0, size-1)
            c_detail_center = random.randint(0, size-1)
            rad_detail = random.randint(1, max(2, size // 40))

            if _can_place_disk(r_detail_center, c_detail_center, rad_detail, size, all_placed_disks):
                rr, cc = disk((r_detail_center, c_detail_center), rad_detail, shape=(size,size))
                phantom[rr,cc] = val_detail
                all_placed_disks.append({'r': r_detail_center, 'c': c_detail_center, 'rad': rad_detail})
                break
                
    return np.clip(phantom, 0, 1.5)

# # Generate the phantom
# generated_phantom_image = generate_phantom()

# # Display the phantom
# plt.figure(figsize=(6,6))
# plt.imshow(generated_phantom_image, cmap='gray')
# plt.colorbar(label='Intensity')
# plt.title(f"Generated Phantom ({PHANTOM_SIZE_XY}x{PHANTOM_SIZE_XY})")
# plt.xlabel("X-coordinate")
# plt.ylabel("Y-coordinate")
# plt.show()