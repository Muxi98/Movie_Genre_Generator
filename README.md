# Movie-Genre-Generator

PyTorch implementation of multi-label movie genre classification from posters comparing Custom CNN and LoRA-wrapped Vision Transformer architectures.

<h1 align="center">
  <!-- Place a sample image or layout asset here once available -->
  <img src="https://github.com/user-attachments/assets/placeholder-hash" height="400" alt="Movie Genre Generator Dashboard">
</h1>
  <p align="center">
    <a>Tomer Biran</a> •
    <a<>Adi Friedlander</a>
  </p>
<p align="center">
Video:
  <a href="#">TBC</a>
</p>

- [Table of contents](#Table-of-contents)
  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Quick start](#Quick-start)
  * [Analysing and testing](#Analysing-and-testing)
  * [Future Work](#Future-Work)
  * [References](#references)

## Background
The objective of this project is to evaluate and solve the complex problem of multi-label classification on highly abstract visual media: movie posters. We analyze and compare two distinct engineering paradigms: a specialized, custom convolutional neural network (**CNN Baseline**) initialized from scratch using Kaiming Normal distribution, versus a heavy foundation network (**Vision Transformer - ViT**) adapted efficiently via Low-Rank Adaptation (**LoRA**) to train specialized low-rank parameter paths (`rank=16`, `alpha=16`) while keeping core foundation weights frozen. 

Due to the sparse nature of multi-label genre assignments, this framework explores the trade-offs between localized translation-invariant features (CNN) and global multi-head self-attention mechanisms (ViT) operating over continuous data streams on resource-constrained execution environments (CPU).

## Dataset
The project utilizes the online streaming layer of the `stzhao/movie_posters_100k_controlnet` dataset hosted via Hugging Face. The data is parsed in real time to yield movie poster matrices mapped across **15 distinct binary genre classification labels**. This setup eliminates the need for massive local storage, streaming batches dynamically directly into training loops.

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

## Files in the repository

| File name | Purpose |
|---|---|
| `run_trains.ipynb` | Main execution notebook managing training loops, real-time RAM metrics tracking, and matplotlib curve plotting. |
| `dataset_pipeline.py` | Houses pipeline utilities to instantiate the real-time internet-streaming PyTorch DataLoader. |
| `train/train_engine.py` | Implements epoch orchestration steps, gradient accumulation handling, and evaluation computation. |
| `utils/cnn.py` | Architecture blueprint configuring the deep Custom CNN built from scratch. |
| `utils/vit.py` | Architecture module injecting Hugging Face PEFT LoRA adapter configurations into the ViT backbone. |

## Quick start

- Clone the repository:
```console
git clone [https://github.com/Muxi98/Movie_Genre_Generator.git](https://github.com/Muxi98/Movie_Genre_Generator.git)
