import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
# =====================================================================
#  MULTI-LABEL CONFUSION MATRICES
# =====================================================================
def generate_multilabel_confusion_matrices(model, dataloader, device, genre_names, optimal_thresholds, save_name="cnn_cm.png"):
    """
    Extracts predictions and graphs individual 2x2 confusion matrices for all genres in a 3x4 grid.
    """
    model.eval()
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_targets.append(labels.numpy())
            
            # Apply the specific threshold for each genre!
            preds = (probs > optimal_thresholds).astype(int)
            all_preds.append(preds)
            
    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)
    
    # Computes an array of 2x2 matrices: one per genre
    mcm = multilabel_confusion_matrix(targets, preds)
    
    # Create a 3x4 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12), dpi=300)
    
    # Flatten the 2D array of axes into a 1D list so we can iterate over it easily
    axes = axes.flatten() 
    
    for idx, genre_name in enumerate(genre_names):
        ax = axes[idx]
        matrix = mcm[idx]
        
        # Formatted 2x2: [[True Negative, False Positive], [False Negative, True Positive]]
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
        
        ax.set_title(f'Genre: {genre_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Class')
        
        # Only add the Y-axis label to the very first column of each row to keep it clean
        if idx % 4 == 0:
            ax.set_ylabel('True Class')
            
    # Optional: If you ever have fewer than 12 genres, hide the empty subplots at the end
    for i in range(len(genre_names), len(axes)):
        fig.delaxes(axes[i])
            
    plt.suptitle("Multi-Label Confusion Matrices (All Categories)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
