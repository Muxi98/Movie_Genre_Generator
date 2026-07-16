import torch
import torch.nn as nn

class MovieHybridModel(nn.Module):
    def __init__(self, trained_cnn, trained_vit, num_classes=12):
        super(MovieHybridModel, self).__init__()
        
        # 1. Isolate the CNN backbone (Everything except the final classifier sequential block)
        self.cnn_features = nn.Sequential(
            trained_cnn.block1,
            trained_cnn.block2,
            trained_cnn.block3,
            trained_cnn.block4,
            trained_cnn.block5,
            trained_cnn.global_pool
        )
        
        # 2. Isolate the ViT backbone (Extract the PEFT LoRA base model directly)
        # This bypasses the original vit classification head automatically
        self.vit_features = trained_vit.vit.vit
        
        # 3. Calculate joint embedding space size:
        # MovieCNN outputs 512 dimensions. vit-base-patch16 hidden size is 768 dimensions.
        combined_dim = 512 + 768  # 1280 dimensions total
        
        # 4. Define the brand-new joint multi-label classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features from your custom CNN
        cnn_out = self.cnn_features(x)
        cnn_out = torch.flatten(cnn_out, 1)  # Shape: [Batch_Size, 512]
        
        # Extract features from your LoRA ViT (Extracting the pooler_output CLS token)
        vit_out = self.vit_features(pixel_values=x)
        vit_out = vit_out.last_hidden_state[:, 0, :]     # Shape: [Batch_Size, 768]
        
        # Concatenate features side-by-side
        combined_features = torch.cat((cnn_out, vit_out), dim=1)  # Shape: [Batch_Size, 1280]
        
        # Output 19 unnormalized logits
        return self.classifier(combined_features)
