import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =====================================================================
#  EXPERIMENTAL HYPERPARAMETER IMPORTANCE ESTIMATOR
# =====================================================================
def plot_hyperparameter_importance(experiment_log_list, output_name="hparam_importance.png"):
    """
    Evaluates hyperparameter impacts using data from multiple validation runs.
    """
    if len(experiment_log_list) < 3:
        print(" Analytics Warning: Hyperparameter evaluation requires a list containing data from at least 3 separate training runs to determine trends.")
        return
        
    df_hparams = pd.DataFrame(experiment_log_list)
    
    # Compute linear correlation matrix against your Target Performance Metric (Validation Macro F1)
    correlations = df_hparams.corr()['final_val_macro_f1'].drop('final_val_macro_f1').sort_values()
    
    plt.figure(figsize=(8, 4), dpi=300)
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in correlations.values]
    
    sns.barplot(x=correlations.values, y=correlations.index, palette=colors)
    plt.title("Hyperparameter Sensitivity Analysis (Correlation to Macro F1-Score)", fontsize=12, fontweight='bold')
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Configured Hyperparameters")
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    plt.xlim(-1, 1)
    plt.tight_layout()
    plt.savefig(output_name, bbox_inches='tight')
    plt.show()
