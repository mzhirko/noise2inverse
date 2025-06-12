import kagglehub
import shutil
import os
import pandas as pd
import os
import shutil

# Download latest version
destination_dir = "./dataset"
base_dir = os.path.join(destination_dir, '11')

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
if not os.path.exists(base_dir):
    path = kagglehub.dataset_download("maedemaftouni/large-covid19-ct-slice-dataset")
    shutil.move(path, destination_dir)
    print(f"Dataset moved from {path} to {destination_dir}")

csv_file_path = os.path.join(base_dir, 'meta_data_normal.csv')
source_image_directory = os.path.join(base_dir, 'curated_data', 'curated_data', '1NonCOVID')

target_folder_name = './CT_dataset'
file_name_column = 'File name'
patient_id_column = 'Patient ID'

df = pd.read_csv(csv_file_path)

# Group by 'Patient ID' and sample 3 file names
# Assumes each patient has at least 3 files as per your problem description
sampled_file_names = df.groupby(patient_id_column)[file_name_column].apply(
    lambda x: x.sample(n=3, random_state=1) # random_state for consistency
).reset_index(drop=True)

# Create the target folder if it doesn't exist
os.makedirs(target_folder_name, exist_ok=True) # exist_ok=True prevents error if dir exists

print(f"Target folder '{target_folder_name}' ensured.")
print(f"Attempting to copy {len(sampled_file_names)} files...")

# Copy the sampled files
for file_name in sampled_file_names:
    base_file_name = os.path.basename(file_name) # Get only the filename part
    source_file_path = os.path.join(source_image_directory, base_file_name)
    destination_file_path = os.path.join(target_folder_name, base_file_name)
    
    shutil.copy(source_file_path, destination_file_path)
    print(f"  Copied: '{source_file_path}' to '{destination_file_path}'")
    
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

print(f"\nSuccessfully copied {len(sampled_file_names)} files to '{target_folder_name}'.")
print("Script finished.")