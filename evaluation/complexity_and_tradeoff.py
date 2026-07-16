import pandas as pd
import torch
import time
import os
# =============================================
# MODEL COMPLEXITY & TRADEOFF ANALYSIS
# =============================================
def compare_model_complexity(models_dict, dataloader, device, experiment_dirs=None):
    """
    Calculates parameter footprints and inference speeds for multiple models,
    and returns a formatted DataFrame ready for a report.
    """
    results = []
    
    # Grab one real batch of images to use for speed benchmarking
    dummy_images, _ = next(iter(dataloader))
    dummy_images = dummy_images.to(device)
    batch_size = dummy_images.size(0)
    
    for name, model in models_dict.items():
        model.eval()
        
        # 1. Parameter Footprint Calculation
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 2. Inference Speed Measurement
        # (A) Warmup step to spin up GPU clocks to max speed
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_images)
                
        # (B) Official timing (using CUDA sync to ensure GPU operations finish before the clock stops)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        # Run 50 passes to get a stable average
        num_passes = 50
        with torch.no_grad():
            for _ in range(num_passes):
                _ = model(dummy_images)
                
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        
        avg_ms_per_batch = ((end_time - start_time) / num_passes) * 1000
        
        # 3. Fetch Best Val Metrics from disk (if provided)
        best_f1, best_loss = None, None
        if experiment_dirs and name in experiment_dirs:
            csv_path = os.path.join(experiment_dirs[name], "metrics_history.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                best_f1 = df['val_macro_f1'].max()
                best_loss = df['val_loss'].min()
                
        # 4. Compile Row
        results.append({
            "Model Architecture": name,
            "Total Params (M)": f"{total_params / 1e6:.2f}M",
            "Trainable Params (M)": f"{trainable_params / 1e6:.2f}M",
            "Trainable %": f"{(trainable_params / total_params) * 100:.3f}%",
            f"Inference Time / Batch ({batch_size})": f"{avg_ms_per_batch:.1f} ms",
            "Best Val Loss": f"{best_loss:.4f}" if best_loss else "N/A",
            "Best Val F1": f"{best_f1:.4f}" if best_f1 else "N/A"
        })
        
    df_results = pd.DataFrame(results)
    
    # Print standard table
    print("\n Model Complexity & Performance Summary:\n")
    print(df_results.to_string(index=False))
    
    # Also print a markdown version so you can easily copy-paste it into your report/README
    print("\n Markdown Version (for easy copy-pasting):")
    print(df_results.to_markdown(index=False))
    
    return df_results
