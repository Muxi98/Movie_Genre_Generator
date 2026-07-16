import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# =============================================
# CLASS IMBALANCE & POS_WEIGHT ANALYSIS
# =============================================
def plot_class_imbalance(parquet_path, genre_names, save_name="class_imbalance.png"):
    """
    Analyzes the frequency of each genre in the dataset, plots a distribution chart,
    and calculates the theoretical pos_weight needed to balance the BCE loss.
    """
    # 1. Load the dataset
    df = pd.read_parquet(parquet_path)
    total_samples = len(df)
    
    # 2. Count the occurrences of each genre
    genre_counts = df[genre_names].sum().sort_values(ascending=False)
    
    # Convert to a DataFrame for plotting
    df_counts = pd.DataFrame({'Genre': genre_counts.index, 'Count': genre_counts.values})
    df_counts['Percentage'] = (df_counts['Count'] / total_samples) * 100
    
    # 3. Plot the class distribution
    plt.figure(figsize=(10, 6), dpi=300)
    ax = sns.barplot(data=df_counts, x='Count', y='Genre', palette='magma')
    
    # Add numerical labels (Count and %) to the bars
    for p, pct in zip(ax.patches, df_counts['Percentage']):
        ax.annotate(f"{int(p.get_width()):,} ({pct:.1f}%)", 
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', 
                    fontsize=10, fontweight='bold', 
                    xytext=(5, 0), textcoords='offset points')
                    
    plt.title(f"Dataset Imbalance: Genre Distribution (Total: {total_samples:,} Movies)", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Occurrences (Positive Labels)")
    plt.ylabel("Movie Genre")
    plt.xlim(0, max(df_counts['Count']) * 1.30) # Leave space for the text labels
    plt.grid(axis='x', alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
    
    # 4. Calculate the BCE pos_weight for the report
    print("\n Calculated 'pos_weight' for BCEWithLogitsLoss:")
    print("(This is the multiplier applied to False Negatives to force the model to learn rare classes)")
    print("-" * 80)
    
    pos_weights_dict = {}
    for genre, count in genre_counts.items():
        neg_count = total_samples - count
        weight = neg_count / count
        pos_weights_dict[genre] = weight
        print(f"  {genre:20s} | Positives: {int(count):>6,} | Negatives: {int(neg_count):>6,} | Weight: {weight:>5.2f}x")
        
    return df_counts, pos_weights_dict
