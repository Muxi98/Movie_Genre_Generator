# =============================================
# ROC & PRECISION-RECALL CURVE ANALYSIS
# =============================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_pr_curves(model, dataloader, device, genre_names, save_name="roc_pr_curves.png"):
    """
    Evaluates the model and plots the ROC and PR curves for every individual class.
    Calculates the Area Under the Curve (AUC) and Average Precision (AP).
    """
    model.eval()
    all_targets, all_probs = [], []
    
    # 1. Run inference to get raw Sigmoid probabilities
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            
    targets = np.vstack(all_targets)
    probs = np.vstack(all_probs)
    
    n_classes = len(genre_names)
    colors = plt.cm.get_cmap('tab20', n_classes) # Distinct colors for 12 lines
    
    # 2. Setup the side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # --- Subplot 1: ROC Curve ---
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=colors(i), lw=2, alpha=0.8,
                     label=f"{genre_names[i]} (AUC = {roc_auc:.2f})")
                     
    # The diagonal "Random Guessing" line
    axes[0].plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontweight='bold')
    axes[0].set_ylabel('True Positive Rate (Recall)', fontweight='bold')
    axes[0].set_title('ROC Curves (Per Class)', fontsize=14, fontweight='bold')
    
    # Sort legend by AUC score (descending)
    handles, labels = axes[0].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('= ')[1], reverse=True))
    axes[0].legend(handles, labels, loc="lower right", fontsize=9)
    
    # --- Subplot 2: Precision-Recall Curve ---
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(targets[:, i], probs[:, i])
        ap = average_precision_score(targets[:, i], probs[:, i])
        axes[1].plot(recall, precision, color=colors(i), lw=2, alpha=0.8,
                     label=f"{genre_names[i]} (AP = {ap:.2f})")
                     
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (True Positive Rate)', fontweight='bold')
    axes[1].set_ylabel('Precision', fontweight='bold')
    axes[1].set_title('Precision-Recall Curves (Per Class)', fontsize=14, fontweight='bold')
    
    # Sort legend by AP score (descending)
    handles, labels = axes[1].get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0].split('= ')[1], reverse=True))
    axes[1].legend(handles, labels, loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
