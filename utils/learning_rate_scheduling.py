import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from sklearn.metrics import f1_score

from utils.logging_utils import JupyterExperimentTracker
from utils.train_utils import evaluate_validation_split

def execute_lr_model_training(model, train_loader, val_loader, optimizer, device, num_epochs, experiment_name, pos_weight, scheduler=None):
    tracker = JupyterExperimentTracker(experiment_name=experiment_name)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    start_epoch = tracker.load_latest_checkpoint(model, optimizer)
    best_val_f1 = max([x['val_macro_f1'] for x in tracker.history]) if tracker.history else 0.0

    for epoch in range(start_epoch, num_epochs + 1):
        # Grab the current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']
        tracker.log(f"--- Starting Epoch {epoch}/{num_epochs} | Current LR: {current_lr:.2e} ---")
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        all_train_targets, all_train_preds = [], []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            all_train_targets.append(labels.cpu().numpy())
            all_train_preds.append((torch.sigmoid(logits) > 0.5).cpu().numpy())
            
            if batch_idx % 100 == 0:
                batch_duration = time.time() - batch_start_time
                images_per_sec = images.size(0) / batch_duration if batch_duration > 0 else 0
                tracker.log(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Velocity: {images_per_sec:.1f} img/sec")
                
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_macro_f1 = f1_score(np.vstack(all_train_targets), np.vstack(all_train_preds), average='macro', zero_division=0)
        
        # Validation Evaluation Step
        avg_val_loss, val_macro_f1 = evaluate_validation_split(model, val_loader, device)
        
        epoch_total_duration = time.time() - epoch_start_time
        tracker.log(f"Epoch {epoch} Done in {epoch_total_duration:.1f}s | Train Loss: {avg_train_loss:.4f} | Val F1: {val_macro_f1:.4f}")
        
        # ==========================================
        # NEW: STEP THE SCHEDULER
        # ==========================================
        if scheduler is not None:
            # ReduceLROnPlateau requires the validation metric it is monitoring
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Add the learning rate to your history dictionary so we can graph it later
        stats = {
            "epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "train_macro_f1": train_macro_f1, "val_macro_f1": val_macro_f1,
            "epoch_time_seconds": epoch_total_duration, "learning_rate": current_lr
        }
        tracker.history.append(stats)
        pd.DataFrame(tracker.history).to_csv(tracker.metrics_history_path, index=False)
        
        is_best = val_macro_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_macro_f1
            
        tracker.save_checkpoint(model, optimizer, epoch, is_best=is_best)

    tracker.log("🏁 Experiment optimization process finalized.")
