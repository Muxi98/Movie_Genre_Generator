import torch
import time
import numpy as np
from sklearn.metrics import f1_score, hamming_loss
def train_one_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=8, max_steps=100):
    """
    Trains the model for one epoch over the streaming dataset using gradient accumulation
    to preserve memory on constrained CPU architectures.
    """
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    batch_loss = 0.0
    start_time = time.time()
    
    print(f"Starting training loop (Max streaming steps set to: {max_steps})...")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move tensors to the designated execution device (CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass through the active model architecture
        outputs = model(images)
        
        # Calculate loss and scale it to match the accumulation footprint
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        
        running_loss += loss.item() * accumulation_steps
        batch_loss += loss.item() * accumulation_steps
        
        # Trigger weight updates once the accumulation threshold is met
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            avg_step_loss = batch_loss / accumulation_steps
            print(f"Step [{(batch_idx + 1) // accumulation_steps}] | "
                  f"Streamed Batches: {batch_idx + 1} | "
                  f"Accumulated Loss: {avg_step_loss:.4f} | "
                  f"Time Elapsed: {time.time() - start_time:.1f}s")
            batch_loss = 0.0
            
        # Safety boundary filter to prevent infinite streaming loops during development
        if batch_idx >= (max_steps - 1):
            print("Reached target execution step boundary limit.")
            break
            
    total_avg_loss = running_loss / (batch_idx + 1)
    return total_avg_loss

def evaluate_model(model, dataloader, device, max_steps=20):
    """
    Evaluates the model on a validation data stream using multi-label metrics.
    All code comments are written in English.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    print(f"Starting evaluation loop (Capped at {max_steps} steps for CPU safety)...")
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_steps:
                break
                
            images = batch[0].to(device)   # Grabs the first item in the list (the tensor of images)
            targets = batch[1].to(device)  # Grabs the second item in the list (the tensor of labels)            
            # Forward pass to get unnormalized logits
            logits = model(images)
            
            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(logits)
            
            # Threshold at 0.5 to get binary multi-label predictions
            preds = (probs > 0.5).int()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    # Concatenate all batches into large arrays
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate specialized multi-label metrics
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    h_loss = hamming_loss(all_targets, all_preds)
    
    print("Evaluation finalized.")
    return {"macro_f1": macro_f1, "hamming_loss": h_loss}