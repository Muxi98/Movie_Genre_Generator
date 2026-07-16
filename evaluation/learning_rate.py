import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================
# COMPARISON: LR SCHEDULER vs. NO LR SCHEDULER
# =============================================
def plot_lr_scheduler():

    sns.set_theme(style="whitegrid")
    df = pd.read_csv("checkpoints/movie_cnn_scheduled/metrics_history.csv")

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot Validation Loss on the left Y-axis
    ax1.plot(df['epoch'], df['val_loss'], color='#d62728', marker='o', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Validation Loss', color='#d62728', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_title("Impact of ReduceLROnPlateau on Validation Loss", fontsize=14, fontweight='bold')

    # Create a right Y-axis for the Learning Rate
    ax2 = ax1.twinx()  
    ax2.plot(df['epoch'], df['learning_rate'], color='#1f77b4', marker='s', linestyle='--', linewidth=2, label='Learning Rate')
    ax2.set_ylabel('Learning Rate (Log Scale)', color='#1f77b4', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_yscale('log') # Log scale is standard for LR plots

    fig.tight_layout()
    plt.savefig("lr_scheduler_plot.png", bbox_inches='tight')
    plt.show()
