# CT Image Reconstruction Experiments

Deep learning experiments for CT image reconstruction using DnCNN and UNet architectures with different numbers of projection views.

## Setup

### Installation

#### Create Environment
```bash
conda create -n cito python=3.9
conda activate cito
```

#### Install PyTorch
Install a PyTorch version according to your OS and the instruction in https://pytorch.org/get-started/locally/

#### Install Other Dependencies
```bash
pip install -r requirements.txt
conda install -y -c astra-toolbox astra-toolbox
```

### Data Preparation
Download CT dataset:
```bash
python download_dataset.py
```
This downloads the Large COVID-19 CT Slice Dataset and creates `./CT_dataset/` with processed images.
In case of insufficient disk quota, download the dataset in https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset/data. Unzip in the same directory and rename the folder to dataset/

## Running Experiments

### Phantom Data Experiments
```bash
# Train DnCNN with 1024 views
python experiment_phantom.py -m DnCNN -v 1024

# Train UNet with 512 views  
python experiment_phantom.py -m UNet -v 512
```

### Real CT Data Experiments
```bash
# Train DnCNN on CT images
python experiment_CT.py -m DnCNN -v 1024 --image_size 256

# Train UNet on CT images
python experiment_CT.py -m UNet -v 512 --image_size 256
```

**Key parameters:**
- `-m`: Model architecture (`DnCNN`, `UNet`)
- `-v`: Number of projection views (`1024`, `512`, `256`)
- `--image_size`: Target image size (256 for 256×256)

## Model Evaluation

Generate comparison grids showing input, reconstruction, and ground truth:
```bash
# Evaluate DnCNN on phantom data
python evaluate.py -p -m DnCNN

# Evaluate UNet on CT data  
python evaluate.py -m UNet
```

Compare DnCNN performance across different view counts:
```bash
# Compare DnCNN models on phantom data
python cross_view.py -p

# Compare DnCNN models on CT data
python cross_view.py
```

To reproduce the experiments shown in report, run the following commands
```bash
python experiment_phantom.py -m DnCNN -v 256
python experiment_phantom.py -m DnCNN -v 512
python experiment_phantom.py -m DnCNN -v 1024

python experiment_phantom.py -m UNet -v 256
python experiment_phantom.py -m UNet -v 512
python experiment_phantom.py -m UNet -v 1024

python experiment_CT.py -m DnCNN -v 256 --image_size 256
python experiment_CT.py -m DnCNN -v 512 --image_size 256
python experiment_CT.py -m DnCNN -v 1024 --image_size 256

python experiment_CT.py -m UNet -v 256 --image_size 256
python experiment_CT.py -m UNet -v 512 --image_size 256
python experiment_CT.py -m UNet -v 1024 --image_size 256

python evaluate.py -m DnCNN -p
python evaluate.py -m DnCNN

python cross_view.py
python cross_view.py -p
```

## Output Structure

```
trained_models/                    # Phantom experiments
├── views_1024/DnCNN/
├── views_512/UNet/
└── views_256/

trained_models_images/             # CT image experiments  
├── views_1024/
└── views_512/

Generated files:
- Evaluation grids: *_comparison_grid.png
- Cross-view comparisons: *_dncnn_comparison_3x3.png  
- Ground truth: *_ground_truth.png
```

## Troubleshooting

**CUDA Out of Memory:** Reduce batch size or use smaller image sizes
**Dataset Not Found:** Run `python download_dataset.py` first
**Model Loading Errors:** Ensure training completed successfully