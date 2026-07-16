import torch
import torch.nn as nn

class MovieCNN(nn.Module):
    """
    Custom CNN Architecture for Multi-Label Movie Poster Classification.

    Architecture:
    - 5 Convolutional blocks, each with Conv2d -> GroupNorm -> GELU -> MaxPool2d
    - Adaptive Average Pooling down to 1x1 to remain resolution-agnostic
    - 2 Fully Connected layers with Dropout to mitigate overfitting

    Filters: 64 -> 128 -> 256 -> 512 -> 512
    Kernel size: 3x3 for all convolutional layers, stride=1, padding=1
    Pooling: 2x2 MaxPool after each convolutional block
    Activation: GELU

    Input dimension: 3 x 224 x 224 (RGB movie poster streams)
    Output dimension: 12 (unnormalized logits for multi-label binary cross-entropy)
    """

    def __init__(self, num_classes=12):
        super(MovieCNN, self).__init__()

        # Block 1: 3 -> 64 channels
        # Spatial reduction: 224x224 -> 112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),  # 8 groups of 8 channels each
            nn.GELU(),
            nn.MaxPool2d(2, 2)   
        )

        # Block 2: 64 -> 128 channels
        # Spatial reduction: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool2d(2, 2)   
        )

        # Block 3: 128 -> 256 channels
        # Spatial reduction: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.MaxPool2d(2, 2)   
        )

        # Block 4: 256 -> 512 channels
        # Spatial reduction: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 512),
            nn.GELU(),
            nn.MaxPool2d(2, 2)   
        )

        # Block 5: 512 -> 512 channels
        # Spatial reduction: 14x14 -> 7x7
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 512),
            nn.GELU(),
            nn.MaxPool2d(2, 2)   
        )

        # Global average pooling: Compresses 7x7 feature maps down to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head outputting 15 raw values (logits)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Initialize weights using Kaiming normal distribution
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten tensor shape to (Batch_Size, 512)
        x = self.classifier(x)
        return x