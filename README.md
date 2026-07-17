# Movie-Genre-Generator

PyTorch implementation of multi-label movie genre classification from posters comparing Custom CNN and LoRA-wrapped Vision Transformer architectures.


</h1>
  <p align="center">
    <a>Tomer Biran</a> •
    <a<>Adi Friedlander</a>
  </p>


- [Table of contents](#Table-of-contents)
  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Key Features](#Key-Features)
  * [Project Structure](#Project-Structure)
  * [Getting Started](#Getting-Started)
  * [Analytics & Evaluation](#Analytics-&-Evaluation)
  * [References](#references)

## Background
This repository contains the code, models, and analytical tools for a comprehensive deep learning project aimed at predicting movie genres strictly from their visual poster representations. 

This project tackles the difficult computer vision challenge of **Multi-Label Classification** in an environment with highly subjective visual language, overlapping semantic categories, and severe dataset imbalances.

## Dataset
The project utilizes the online streaming layer of the `stzhao/movie_posters_100k_controlnet` dataset hosted via Hugging Face. The data is parsed in real time to yield movie poster matrices mapped across **12 distinct binary genre classification labels**. This setup eliminates the need for massive local storage, streaming batches dynamically directly into training loops.

## Prerequisites
| Library | Version |
|---|---|
| `Python` | `3.11+` |
| `torch` | `2.1.1+` |
| `torchvision` | `0.16.1+` |
| `transformers` | `4.42.3+` |
| `scikit-learn` | `1.3.0+` |
| `matplotlib` | `3.7.2+` |
| `numpy` | `1.23.5+` |
| `pillow` | `10.0.0+` |



## Key Features

* **Custom CNN Architecture**: A lightweight, hierarchical convolutional neural network built from scratch, utilizing modern regularization techniques like GELU activations, Batch Normalization, and Dropout.
* **Pre-trained Vision Transformer (ViT)**: Utilizes a massive HuggingFace `vit-base-patch16-224` backbone.
* **Low-Rank Adaptation (LoRA)**: Fine-tunes the ViT using PEFT (Parameter-Efficient Fine-Tuning) to adapt the attention Query and Value projections to the movie domain while saving VRAM and preventing representation collapse.
* **Hybrid Fusion Network**: A novel architecture that concatenates the local feature embeddings of the frozen CNN with the global contextual embeddings of the frozen ViT, passing them through a shared multi-layer perceptron (MLP) classification head.
* **Imbalance Compensation**: Dynamically calculates per-genre ratios to weight the `BCEWithLogitsLoss` criterion, forcing the network to penalize false negatives in severely underrepresented genres (e.g., *War*, *Western*).
* **Explainable AI (XAI)**: Includes a pure PyTorch implementation of Grad-CAM (Gradient-weighted Class Activation Mapping) to extract visual attention heatmaps and prove the models have acquired genuine semantic reasoning.

---

## Project Structure

```text
Deep/Project/
├── project_notebook.ipynb         # The main execution hub (Training, Analytics, and Visualizations)
├── environment.yml                # Conda environment dependencies
├── metadata_one_hot_12genres.parquet # Cleaned and reduced dataset (12 core genres)
├── Posters/                       # Directory containing raw movie poster image files
│
├── data/
│   └── dataloader.py              # PyTorch Dataset definitions and image transformation pipelines
│
├── models/
│   ├── cnn_architecture.py        # Baseline MovieCNN architecture
│   ├── vit_architecture.py        # MovieViT with injected LoRA configurations
│   └── hybrid.py                  # Fused architecture joining frozen CNN and ViT embeddings
│
├── training/
│   └── initialize_datasets.py     # Train/Val/Test splitting and DataLoader instantiation
│
├── utils/
│   ├── train_utils.py             # Standard execution loops for model training and validation
│   ├── learning_rate_scheduling.py# Advanced training loop supporting ReduceLROnPlateau
│   └── logging_utils.py           # Experiment tracker for saving metrics, hyperparams, and model checkpoints
│
├── evaluation/                    # Extensive suite of analytical tools and metrics
│   ├── ROC.py                     # Generates per-class ROC and Precision-Recall Curves
│   ├── augmentation.py            # Analyzes Generalization Gap between augmented/unaugmented runs
│   ├── class_performance.py       # Sorts and plots F1-scores by individual genre
│   ├── class_weight_imbalance.py  # Calculates loss weights based on label scarcity
│   ├── complexity_and_tradeoff.py # Benchmarks parameter footprints and inference speeds
│   ├── confusion_matrices.py      # Multi-label 2x2 confusion matrix grid generation
│   ├── explainable.py             # Pure PyTorch Grad-CAM engine and heatmap overlay tools
│   ├── learning_rate.py           # Plots the impact of dynamic LR schedulers against validation loss
│   ├── model_analytics.py         # Baseline loss curves and epoch metrics
│   ├── overfit_generalize.py      # Tracks Train vs Val F1 gaps to monitor overfitting trajectories
│   ├── SNR.py                     # Accuracy analysis against active label densities
│   └── threshold_sensitivity.py   # Sweeps decision boundaries to mathematically optimize Macro F1
│
└── checkpoints/                   # Auto-generated directory tracking model weights and history
```

---

## Getting Started

### 1. Environment Setup
The project relies on standard deep learning libraries (`torch`, `torchvision`, `transformers`, `peft`, `sklearn`, `pandas`, `opencv-python`). 
You can recreate the exact conda environment used for this project:
```bash
conda env create -f environment.yml
conda activate <env_name>
```

### Execution
The easiest way to explore the project is by running the cells in `project_notebook.ipynb`. 
1. **Data Loading**: It automatically compiles the PyTorch datasets.
2. **Experiment Execution**: Run the training cells for the CNN, ViT, or Hybrid models.
3. **Analytics Generation**: The notebook calls the modules inside the `evaluation/` directory to automatically generate and save high-quality graphs and metrics to your local folder.

---

## Analytics & Evaluation
Standard multi-class accuracy is a flawed metric for multi-label datasets. This project relies heavily on rigorous metrics:
* **Threshold Sweeping**: Finding the optimal Sigmoid cutoff boundary rather than relying on a default `0.5`.
* **Precision-Recall (AP)**: Utilized heavily to ensure models aren't cheating by predicting "absent" for minority classes (which artificially inflates ROC AUC scores).
* **Attention Heatmaps**: Used systematically across all genres for single images to visualize how the network's focal points shift depending on the target genre.

---

## References

[1] https://github.com/sbajamy/Music-Genre-Classification-Using-Audio-Spectrogram-Transformer

[2] https://huggingface.co/datasets/stzhao/movie_posters_100k_controlnet

[3] https://github.com/Spandan-Madan/DeepLearningProject/tree/master

[4] https://www.kaggle.com/code/krutarthhd/genre-prediction-from-the-movie-poster

[5] https://github.com/ovedtal1/MambaVision-Genre-Classification
