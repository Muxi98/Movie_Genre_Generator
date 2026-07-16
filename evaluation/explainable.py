# =============================================
# EXPLAINABLE AI: PURE PYTORCH GRAD-CAM
# =============================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

class PureGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to intercept the data flowing through the CNN
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output # Save the feature maps
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] # Save the gradients
        
    def __call__(self, x, class_idx):
        self.model.eval()
        
        # 1. Forward pass
        logits = self.model(x)
        
        # 2. Backward pass for the SPECIFIC genre we want to explain
        self.model.zero_grad()
        target_logit = logits[0, class_idx] 
        target_logit.backward(retain_graph=True)
        
        # 3. Pull intercepted data off the GPU
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 4. Global Average Pooling to find the "importance" of each feature map channel
        weights = np.mean(gradients, axis=(1, 2))
        
        # 5. Multiply the feature maps by their importance weights
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 6. ReLU (we only care about features that POSITIVELY influenced the genre)
        cam = np.maximum(cam, 0)
        
        # 7. Normalize between 0 and 1 for rendering
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam

def plot_heatmap(image_tensor, cam, target_genre, true_genres=None, save_name=None):
    """Takes the raw tensor and the CAM mask and plots the overlay."""
    cam_resized = cv2.resize(cam, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    overlay = 0.5 * heatmap + 0.5 * img
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    
    # --- NEW: Dynamically build the title ---
    title_str = f"Grad-CAM Explanation for predicting: '{target_genre}'"
    if true_genres:
        title_str += f"\n(Actual Movie Genres: {true_genres})"
        
    fig.suptitle(title_str, fontsize=13, fontweight='bold', y=1.12)
    # ----------------------------------------
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title("Raw Attention Map")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red = High Attention)")
    axes[2].axis('off')
    
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()


# def plot_heatmap(image_tensor, cam, genre_name, save_name=None):
#     """Takes the raw tensor and the CAM mask and plots the overlay."""
#     # Resize the tiny heatmap back to 224x224
#     cam_resized = cv2.resize(cam, (224, 224))
    
#     # Create the Red/Yellow/Blue heatmap colors
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = heatmap[..., ::-1] # BGR to RGB
    
#     # Un-normalize the original image tensor so humans can see it
#     img = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = std * img + mean
#     img = np.clip(img, 0, 1)
    
#     # Overlay the heatmap on the image
#     overlay = 0.5 * heatmap + 0.5 * img
    
#     # Plotting
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
#     fig.suptitle(f"Grad-CAM Explanation for predicting: '{genre_name}'", fontsize=14, fontweight='bold', y=1.05)
    
#     axes[0].imshow(img)
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')
    
#     axes[1].imshow(cam_resized, cmap='jet')
#     axes[1].set_title("Raw Attention Map")
#     axes[1].axis('off')
    
#     axes[2].imshow(overlay)
#     axes[2].set_title("Overlay (Red = High Attention)")
#     axes[2].axis('off')
    
#     if save_name:
#         plt.savefig(save_name, bbox_inches='tight')
#     plt.show()
