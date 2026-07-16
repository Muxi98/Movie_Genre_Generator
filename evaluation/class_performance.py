import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
# =============================================
# PER-CLASS PERFORMANCE ANALYSIS
# =============================================
def plot_per_class_f1(model, dataloader, device, genre_names, threshold=0.5, save_name="per_class_f1.png"):
    """
    Evaluates the model on the provided dataloader and plots a sorted horizontal 
    bar chart of the F1-Score for every individual class.
    """
    model.eval()
    all_targets, all_preds = [], []
    
    # 1. Run full inference on the dataset
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_targets.append(labels.numpy())
            all_preds.append((probs.cpu().numpy() > threshold).astype(int))
            
    targets = np.vstack(all_targets)
    preds = np.vstack(all_preds)
    
    # 2. Get the detailed classification report from sklearn
    report = classification_report(
        targets, preds, target_names=genre_names, 
        output_dict=True, zero_division=0
    )
    
    # 3. Extract just the F1-scores for the individual genres
    f1_scores = {genre: report[genre]['f1-score'] for genre in genre_names}
    
    # 4. Convert to a DataFrame and sort from best to worst
    df_f1 = pd.DataFrame(list(f1_scores.items()), columns=['Genre', 'F1-Score'])
    df_f1 = df_f1.sort_values('F1-Score', ascending=False)
    
    # 5. Plot the results
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(data=df_f1, x='F1-Score', y='Genre', palette='viridis')
    
    # Add the exact numerical values next to the bars
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", 
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', 
                    fontsize=10, fontweight='bold', 
                    xytext=(5, 0), textcoords='offset points')
                    
    plt.title("Per-Class F1-Score Analysis (CNN Model)", fontsize=14, fontweight='bold')
    plt.xlabel("F1-Score (Higher is better)")
    plt.ylabel("Movie Genre")
    plt.xlim(0, max(df_f1['F1-Score']) + 0.15) # Leave space for the text labels
    plt.grid(axis='x', alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
    # Print the full raw report for your own inspection
    print("\n Full Detailed Classification Report:")
    print(classification_report(targets, preds, target_names=genre_names, zero_division=0))
    
    return df_f1
