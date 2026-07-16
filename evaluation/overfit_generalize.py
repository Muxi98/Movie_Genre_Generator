import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# =============================================
# OVERFITTING & GENERALIZATION GAP ANALYSIS
# =============================================
def plot_overfitting_gap(experiment_dirs, save_name="overfitting_gap.png"):
    """
    Plots the Generalization Gap (Train F1 - Val F1) over epochs to measure overfitting.
    Args:
        experiment_dirs (dict): Dictionary mapping model names to their checkpoint folders.
                                e.g., {"MovieCNN": "checkpoints/movie_cnn_experiment"}
    """
    plt.figure(figsize=(9, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # Generate distinct colors for each model
    colors = sns.color_palette("Set1", len(experiment_dirs))
    
    for idx, (model_name, dir_path) in enumerate(experiment_dirs.items()):
        csv_path = os.path.join(dir_path, "metrics_history.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # 1. Calculate the Overfitting Gap (Train F1 minus Val F1)
            # If the gap is positive and growing, the model is overfitting heavily.
            df['f1_gap'] = df['train_macro_f1'] - df['val_macro_f1']
            
            # 2. Plot the trajectory
            plt.plot(df['epoch'], df['f1_gap'], label=model_name, 
                     marker='o', linewidth=2.5, color=colors[idx])
        else:
            print(f"⚠️ Warning: Could not find metrics for {model_name} at {csv_path}")
            
    plt.title("Overfitting Trajectory: The Generalization Gap (Train F1 - Val F1)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Epochs")
    plt.ylabel("F1 Gap Magnitude (Lower is better)")
    
    # Add a dashed line at y=0 representing a mathematically "Perfect" model
    plt.axhline(0, color='black', linestyle='--', alpha=0.6, label='Perfect Generalization (Zero Gap)')
    
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight')
    plt.show()
