import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model

class MovieViT(nn.Module):
    """
    Vision Transformer (ViT) wrapped with Low-Rank Adaptation (LoRA) 
    for resource-efficient multi-label movie poster classification.
    All code comments are written in English.
    """
    def __init__(self, num_classes=12, rank=16, alpha=16, dropout=0.3):
        super(MovieViT, self).__init__()
        
        # 1. Load the pre-trained backbone base model from Hugging Face
        base_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            problem_type="multi_label_classification"
        )
        
        # 2. Dynamically scan the layers to find the correct attention keys automatically
        detected_targets = []
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear) and any(k in name for k in ["query", "value", "q_proj", "v_proj"]):
                leaf_name = name.split(".")[-1]
                if leaf_name not in detected_targets:
                    detected_targets.append(leaf_name)
                    
        print(f"PEFT Target Alignment: Automatically targeting modules: {detected_targets}")
        
        # 3. Define the parameter-efficient fine-tuning configuration
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=detected_targets,
            lora_dropout=dropout,
            bias="none",
            modules_to_save=["classifier"]  # Ensures classification linear head stays active
        )
        
        # 4. Wrap the base model with the PEFT optimization layers
        self.vit = get_peft_model(base_model, lora_config)

    def forward(self, x):
        # Hugging Face models return custom sequence objects; extract the raw logits
        outputs = self.vit(pixel_values=x)
        return outputs.logits

    def print_trainable_parameters(self):
        """
        Displays the parameter footprint reduction achieved via LoRA adaptation.
        """
        self.vit.print_trainable_parameters()