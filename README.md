# VGG16 and ResNet50 Image Classification

Multi-class image classification using transfer learning with VGG16 and ResNet50.

## Dataset structure

Place images in the following folders:

- `dataset/train/cats/`
- `dataset/train/dogs/`
- `dataset/train/horses/`
- `dataset/validation/cats/`
- `dataset/validation/dogs/`
- `dataset/validation/horses/`

Use RGB images; the models will resize images to 224 × 224 pixels. Keep training and validation images separate.

## Project folders

- `notebooks/` — Jupyter notebook for the assignment
- `outputs/plots/` — training and evaluation plots
- `outputs/confusion_matrices/` — confusion matrix images
- `outputs/results/` — saved evaluation results
- `test_images/` — new images for prediction
- `main.py` — Python entry point

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

The final notebook and report will be added after training and evaluation.
