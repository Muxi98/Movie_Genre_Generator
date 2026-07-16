import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
# =============================================
# THRESHOLD SENSITIVITY ANALYSIS
# =============================================
def plot_threshold_sensitivity(model, dataloader, device, save_name="threshold_sensitivity.png"):
    """
    Evaluates the model across a range of decision boundaries (0.05 to 0.95) to find
    the optimal threshold that maximizes the Macro F1-Score.
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
    
    # 2. Sweep thresholds from 0.05 to 0.95
    thresholds = np.arange(0.05, 0.96, 0.05)
    f1_scores, precisions, recalls = [], [], []
    
    for t in thresholds:
        preds = (probs > t).astype(int)
        
        # Calculate metrics using macro average
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        p = precision_score(targets, preds, average='macro', zero_division=0)
        r = recall_score(targets, preds, average='macro', zero_division=0)
        
        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        
    # 3. Find the optimal threshold mathematically
    max_f1 = max(f1_scores)
    optimal_t = thresholds[np.argmax(f1_scores)]
    
    # 4. Plot the Precision-Recall tradeoff curves
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # Plot the 3 lines
    plt.plot(thresholds, f1_scores, label='Macro F1-Score', color='#2ca02c', linewidth=3, marker='o')
    plt.plot(thresholds, precisions, label='Macro Precision', color='#1f77b4', linewidth=2, linestyle='--')
    plt.plot(thresholds, recalls, label='Macro Recall', color='#d62728', linewidth=2, linestyle='--')
    
    # Highlight the optimal sweet spot
    plt.axvline(optimal_t, color='black', linestyle=':', alpha=0.7)
    plt.text(optimal_t + 0.02, max_f1 + 0.02, f"Optimal Threshold: {optimal_t:.2f}\nMax F1: {max_f1:.3f}", 
             fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
    
    plt.title("Threshold Sensitivity & Precision-Recall Tradeoff", fontsize=14, fontweight='bold')
    plt.xlabel("Decision Threshold Boundary (Sigmoid Probability)")
    plt.ylabel("Performance Score (0.0 to 1.0)")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=10, loc='lower center')
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
    print(f"Analysis Complete: The optimal decision boundary is {optimal_t:.2f} (Max F1 = {max_f1:.3f})")
    print(f"Your previous F1-Score at the default 0.50 threshold was roughly {f1_scores[9]:.3f}")
    
    return optimal_t, max_f1