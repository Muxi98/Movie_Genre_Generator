import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================
# COMPARISON: AUGMENTATION vs. NO AUGMENTATION
# =============================================
def compare_augmentation_vs_no_augmentation():
    sns.set_theme(style="whitegrid")
    
    # Load metrics from both experiments
    baseline_csv = "checkpoints/cnn_no_augmentation/metrics_history.csv"
    augmented_csv = "checkpoints/cnn_with_augmentation/metrics_history.csv"

    df_baseline = pd.read_csv(baseline_csv)
    df_augmented = pd.read_csv(augmented_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # ---- Left Plot: Loss Comparison ----
    ax1 = axes[0]
    ax1.plot(df_baseline['epoch'], df_baseline['train_loss'], label='No Aug – Train Loss',
         color='#1f77b4', marker='o', linewidth=2)
    ax1.plot(df_baseline['epoch'], df_baseline['val_loss'], label='No Aug – Val Loss',
         color='#1f77b4', linestyle='--', marker='s', linewidth=2)
    ax1.plot(df_augmented['epoch'], df_augmented['train_loss'], label='With Aug – Train Loss',
         color='#ff7f0e', marker='o', linewidth=2)
    ax1.plot(df_augmented['epoch'], df_augmented['val_loss'], label='With Aug – Val Loss',
         color='#ff7f0e', linestyle='--', marker='s', linewidth=2)

    ax1.set_title('Loss: Augmentation vs. No Augmentation', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Cross-Entropy Loss')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # ---- Right Plot: Macro F1 Comparison ----
    ax2 = axes[1]
    ax2.plot(df_baseline['epoch'], df_baseline['train_macro_f1'], label='No Aug – Train F1',
         color='#2ca02c', marker='o', linewidth=2)
    ax2.plot(df_baseline['epoch'], df_baseline['val_macro_f1'], label='No Aug – Val F1',
         color='#2ca02c', linestyle='--', marker='s', linewidth=2)
    ax2.plot(df_augmented['epoch'], df_augmented['train_macro_f1'], label='With Aug – Train F1',
         color='#d62728', marker='o', linewidth=2)
    ax2.plot(df_augmented['epoch'], df_augmented['val_macro_f1'], label='With Aug – Val F1',
         color='#d62728', linestyle='--', marker='s', linewidth=2)

    ax2.set_title('Macro F1: Augmentation vs. No Augmentation', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1-Score')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    plt.suptitle("Data Augmentation Impact Analysis on MovieCNN",
             fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig("augmentation_comparison.png", bbox_inches='tight')
    plt.show()

    # ---- Print Final Metrics Table ----
    summary = pd.DataFrame({
        'Metric': ['Best Val Loss', 'Best Val Macro F1'],
        'No Augmentation': [
            df_baseline['val_loss'].min(),
            df_baseline['val_macro_f1'].max()
        ],
        'With Augmentation': [
            df_augmented['val_loss'].min(),
            df_augmented['val_macro_f1'].max()
        ]
    })
    summary['Delta'] = summary['With Augmentation'] - summary['No Augmentation']
    print("\n Final Comparison Summary:")
    print(summary.to_string(index=False))
