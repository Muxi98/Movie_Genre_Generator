import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report, precision_score, recall_score
import re

# =====================================================================
#  GRAPH LOSS VS EPOCHS (TRAIN & VAL FOR ALL MODELS)
# =====================================================================
def plot_comparative_loss_curves(experiment_dirs):
    """
    Plots Train vs Validation Loss across epochs for all evaluated models side-by-side.
    Args:
        experiment_dirs (dict): e.g., {"MovieCNN": "checkpoints/movie_cnn_experiment", ...}
    """
    sns.set_theme(style="whitegrid")
    num_models = len(experiment_dirs)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 5), sharey=True, dpi=300)
    
    # Force axes to be an array if there's only 1 model
    if num_models == 1:
        axes = [axes]
        
    for idx, (model_name, dir_path) in enumerate(experiment_dirs.items()):
        csv_path = os.path.join(dir_path, "metrics_history.csv")
        ax = axes[idx]
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            ax.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', marker='o', linewidth=2)
            ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linestyle='--', marker='s', linewidth=2)
            
            ax.set_title(f'{model_name} Learning Curve', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epochs')
            if idx == 0:
                ax.set_ylabel('Binary Cross-Entropy Loss')
            ax.legend()
            ax.grid(True, alpha=0.4)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{csv_path}", ha='center', va='center')
            
    plt.suptitle("Convergence Analysis: Train vs. Validation Loss Profiles", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("analytics_loss_curves.png", bbox_inches='tight')
    plt.show()

# =====================================================================
#  PLOT TRAINING METRICS
# =====================================================================
def plot_training_metrics(log_file_path):
    epochs = []
    train_losses = []
    val_f1_scores = []
    
    # Regex to match lines like:
    # "Epoch 1 Done in 194.3s | Train Loss: 1.0036 | Val F1: 0.3703"
    pattern = re.compile(r"Epoch (\d+) Done.*?Train Loss: ([\d\.]+).*?Val F1: ([\d\.]+)")
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_f1_scores.append(float(match.group(3)))
                
    if not epochs:
        print("No valid epoch data found in the log file.")
        return

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Training Loss on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create a secondary y-axis for Val F1 Score
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Val F1 Score', color=color)  
    ax2.plot(epochs, val_f1_scores, color=color, marker='s', label='Val F1')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add legends and title
    fig.suptitle('Training Loss and Validation F1 vs. Epochs', fontsize=14)
    fig.tight_layout()  
    plt.show()

# =====================================================================
#  PLOT OPTIMAL THRESHOLDS
# =====================================================================
def find_optimal_thresholds(model, val_loader, device, num_classes):
    model.eval()
    all_targets = []
    all_probs = []
    
    # 1. Get all probabilities and targets from the validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    
    optimal_thresholds = []
    
    # 2. For each genre, test thresholds from 0.1 to 0.9 to see which gets the best F1 Score
    for i in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0.0
        
        # Test thresholds: 0.1, 0.2, ... 0.9
        for thresh in np.arange(0.1, 0.95, 0.05):
            preds = (probs[:, i] > thresh).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        optimal_thresholds.append(best_thresh)
        print(f"Class {i} Optimal Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
        
    return np.array(optimal_thresholds)
