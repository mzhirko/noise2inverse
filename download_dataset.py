import kagglehub
import shutil
import os
import pandas as pd

# This script downloads a specific version of the COVID-19 CT slice dataset from Kaggle,
# selects a subset of images, and copies them to a new directory for use.

# --- Configuration ---
# Directory to store the downloaded dataset temporarily.
destination_dir = "./dataset"
# Specific version directory within the dataset.
base_dir = os.path.join(destination_dir, '11')
# The final directory where the selected CT images will be stored.
target_folder_name = './CT_dataset'

# --- Download and Extract Dataset ---
# Ensure the destination directories exist.
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
if not os.path.exists(base_dir):
    # Download the dataset from Kaggle Hub.
    path = kagglehub.dataset_download("maedemaftouni/large-covid19-ct-slice-dataset")
    # Move the downloaded content to the specified destination directory.
    shutil.move(path, destination_dir)
    print(f"Dataset moved from {path} to {destination_dir}")

# --- Image Selection ---
# Paths to the metadata CSV and the source images.
csv_file_path = os.path.join(base_dir, 'meta_data_normal.csv')
source_image_directory = os.path.join(base_dir, 'curated_data', 'curated_data', '1NonCOVID')

# Column names in the CSV file.
file_name_column = 'File name'
patient_id_column = 'Patient ID'

# Read the metadata CSV into a pandas DataFrame.
df = pd.read_csv(csv_file_path)

# Group the DataFrame by patient ID and sample 3 file names for each patient.
# Using a fixed random_state ensures that the same files are selected every time the script is run.
sampled_file_names = df.groupby(patient_id_column)[file_name_column].apply(
    lambda x: x.sample(n=3, random_state=1)
).reset_index(drop=True)

# --- Copy Selected Files ---
# Create the target folder for the selected images if it doesn't already exist.
os.makedirs(target_folder_name, exist_ok=True)

print(f"Target folder '{target_folder_name}' ensured.")
print(f"Attempting to copy {len(sampled_file_names)} files...")

# Iterate through the list of sampled file names and copy them to the target directory.
for file_name in sampled_file_names:
    base_file_name = os.path.basename(file_name)
    source_file_path = os.path.join(source_image_directory, base_file_name)
    destination_file_path = os.path.join(target_folder_name, base_file_name)
    
    shutil.copy(source_file_path, destination_file_path)
    print(f"  Copied: '{source_file_path}' to '{destination_file_path}'")

# --- Cleanup ---
# Remove the temporary dataset directory to save space.
if os.path.exists(destination_dir):
    shutil.rmtree(destination_dir)

print(f"\nSuccessfully copied {len(sampled_file_names)} files to '{target_folder_name}'.")
print("Script finished.")