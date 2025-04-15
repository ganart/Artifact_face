# Artifact Detection in Images

This project implements a deep learning solution for binary image classification to detect artifacts in images. It uses three models—ResNet50, EfficientNetV2-S, and Vision Transformer (ViT)—combined in an ensemble to achieve high performance. The models are trained and evaluated on a custom dataset, achieving an F1 score of 0.97 on the test set.

The main code is executed via a Jupyter Notebook (`hovorukha_task_1.ipynb`), which includes dataset preparation, model training, and evaluation.

## Solution Overview

The solution consists of the following components:

1. **Dataset Preparation**:
   - Dataset is downloaded from Google Drive and split into training, validation, and test sets.
   - Images are labeled as `0` (artifact) or `1` (no artifact).
   - Data augmentation (random cropping, flipping, rotation, color jittering) is applied to the training set.

2. **Models**:
   - **ResNet50**: Pre-trained CNN (ImageNet), fine-tuned for binary classification.
   - **EfficientNetV2-S**: Lightweight pre-trained CNN, fine-tuned for efficiency.
   - **Vision Transformer (ViT)**: Pre-trained transformer (ImageNet), adapted for binary classification.
   - Each model outputs two classes.

3. **Training**:
   - Models are trained using the Adam optimizer and CrossEntropyLoss.
   - Validation is performed after each epoch, with early stopping based on the validation F1 score.

4. **Ensemble**:
   - Combines predictions from all three models by averaging probabilities for the positive class (no artifact).
   - Uses a threshold of 0.5 for binary predictions.
   - Improves robustness and performance over individual models.

5. **Evaluation**:
   - Models are evaluated on the test set using loss and micro-averaged F1 score.
   - Misclassified images are logged with paths, true labels, and predicted labels.

## Final Metrics

| Model                | Test F1 Score |
|----------------------|---------------|
| ResNet50             | 0.900         |
| EfficientNetV2-S     | 0.925         |
| Vision Transformer   | 0.950         |
| Ensemble (All Models)| **0.970**     |

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU (recommended for faster training and inference)
- Internet connection (for downloading dataset and pre-trained weights)
- Jupyter Notebook (for running the main code)



## Installation

## Installation

The main code is executed via a Jupyter Notebook (`hovorukha_task_1.ipynb`), which handles dataset preparation, model training, and evaluation. By default, training functions (`fit`) are commented out, and pre-trained model weights are downloaded from Google Drive, enabling quick accuracy evaluation without training. The notebook is configured for Kaggle, with paths tied to the Kaggle environment.

Follow these steps to set up and run the project:

1. Open `hovorukha_task_1.ipynb` in the Kaggle Notebook environment.

2. Run all cells sequentially, **except** for those marked with:
   ```python
   # EDA: Visualize dataset
These cells are optional and used only for dataset visualization (e.g., displaying sample images). Skipping them will not affect model performance or evaluation.

3. If you want to train models from scratch, uncomment the fit function calls. Otherwise, the notebook will load pre-trained weights and perform immediate evaluation.
