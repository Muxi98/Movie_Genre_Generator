import os
import sys
import logging
import torch
import pandas as pd

class JupyterExperimentTracker:
    def __init__(self, experiment_name, checkpoint_dir="checkpoints"):
        self.experiment_name = experiment_name
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prevent duplicated logging output inside notebook execution cells
        logger = logging.getLogger(experiment_name)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        
        # Saves logs permanently to disk
        file_handler = logging.FileHandler(os.path.join(self.checkpoint_dir, "training.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Streams logs to notebook output screen
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        self.logger = logger
        self.metrics_history_path = os.path.join(self.checkpoint_dir, "metrics_history.csv")
        self.history = []

    def log(self, message):
        self.logger.info(message)

    def record_epoch_stats(self, epoch, train_loss, val_loss, train_macro_f1, val_macro_f1, duration_seconds):
        stats = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_macro_f1": train_macro_f1, "val_macro_f1": val_macro_f1,
            "epoch_time_seconds": duration_seconds
        }
        self.history.append(stats)
        pd.DataFrame(self.history).to_csv(self.metrics_history_path, index=False)

    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'metrics_history': self.history
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, "latest_checkpoint.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, "best_model.pt"))
            self.log(f"Saved new best model configuration to disk!")

    def load_latest_checkpoint(self, model, optimizer):
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if not os.path.exists(latest_path):
            self.log("No previous checkpoint detected. Starting optimization from scratch.")
            return 1 
        self.log(f"Resuming pipeline execution from active checkpoint file...")
        checkpoint = torch.load(latest_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['metrics_history']
        return checkpoint['epoch'] + 1
