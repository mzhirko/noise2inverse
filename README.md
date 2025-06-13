# CT Image Reconstruction Experiments

Deep learning experiments for CT image reconstruction using DnCNN and UNet architectures with different numbers of projection views.

## Setup

### Installation
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data Preparation
Download CT dataset:
```bash
python download_dataset.py
```
This downloads the Large COVID-19 CT Slice Dataset and creates `./CT_dataset/` with processed images.

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

### Hyperparameter Optimization
```bash
python hpo.py
```
Runs automated hyperparameter search for DnCNN with random sampling over learning rates, batch sizes, and network parameters.

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