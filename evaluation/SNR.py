import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
# =====================================================================
#  SIGNAL-TO-NOISE RATIO (SNR) METRIC ACCURACY ANALYSIS
# =====================================================================
def plot_snr_accuracy_distribution(model, dataloader, device, genre_names, save_name="snr_analysis.png"):
    """
    Analyzes model accuracy based on label density (Signal-to-Noise Ratio).
    'Signal' = Number of active genres on a poster.
    'Noise'  = Number of inactive slots out of 19 total.
    """
    model.eval()
    all_targets, all_preds = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.numpy())
            all_preds.append((probs.cpu().numpy() > 0.5).astype(int))
            
    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)
    
    # Signal calculation: count how many labels are True per image row
    active_label_counts = targets.sum(axis=1).astype(int)
    
    # Calculate exact matching subset accuracy for each specific row
    # In multi-label, a row is accurate only if all 19 elements match perfectly
    row_perfect_matches = (targets == preds).all(axis=1).astype(int)
    
    # Group results by structural density
    df_snr = pd.DataFrame({'Active_Labels': active_label_counts, 'Perfect_Match': row_perfect_matches})
    snr_stats = df_snr.groupby('Active_Labels')['Perfect_Match'].agg(['mean', 'count']).reset_index()
    
    # Convert mean to percentage accuracy
    snr_stats['Accuracy_Pct'] = snr_stats['mean'] * 100
    
    plt.figure(figsize=(9, 5), dpi=300)
    ax = sns.barplot(data=snr_stats, x='Active_Labels', y='Accuracy_Pct', palette='crest')
    
    # Add data counters above the bars to inform report data density
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='center', fontsize=9, fontweight='bold', xytext=(0, 5), textcoords='offset points')
                    
    plt.title("XAI Evaluation: Exact Subset Accuracy vs. Number of Active Genres per Poster", fontsize=12, fontweight='bold')
    plt.xlabel("Signal Vector Strength (Count of True Genres Present simultaneously)")
    plt.ylabel("Perfect Row Overlap Match Rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
