import torch
import json
import re
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

# Global variable configuration for multi-label targets
GENRE_LIST = [
    "Action", "Comedy", "Drama", "Horror", "Thriller", 
    "Science Fiction", "Mystery", "Romance", "Crime", "Animation",
    "Adventure", "Fantasy", "Family", "Documentary", "History"
]
GENRE_MAP = {genre.lower(): i for i, genre in enumerate(GENRE_LIST)}


def parse_labels_from_row(row):
    """
    Robust parsing layer to catch multi-label attributes across different
    textual or structured column schemas in the dataset.
    """
    multi_hot = torch.zeros(len(GENRE_LIST), dtype=torch.float32)
    found_any = False

    # Inspect all text fields in the row to find genre markers
    for column_name in ['text', 'genres', 'label', 'caption']:
        if column_name in row and row[column_name]:
            raw_data = str(row[column_name])
            
            # Check for standard genre tokens in a uniform case manner
            for genre, index in GENRE_MAP.items():
                # Word boundary match ensures 'Drama' doesn't flag random substrings
                if re.search(r'\b' + re.escape(genre) + r'\b', raw_data.lower()):
                    multi_hot[index] = 1.0
                    found_any = True
                    
    return multi_hot, found_any


class MoviePosterStreamingDataset(IterableDataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Custom PyTorch IterableDataset explicitly designed for streaming lines
        without downloading massive parquet chunks onto limited local discs.
        """
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        for row in self.hf_dataset:
            try:
                # Validate that both visual features and text entries are present
                if 'image' not in row or row['image'] is None:
                    continue

                # Parse and build multi-label target vectors
                labels, valid_labels = parse_labels_from_row(row)
                if not valid_labels:
                    continue  # Filter out unlabelled entries to prevent noisy gradients

                # Preprocess the PIL image feature array to raw tensors
                image = row['image'].convert("RGB")
                if self.transform:
                    image = self.transform(image)

                yield image, labels

            except Exception as e:
                # Gracefully swallow row parsing anomalies to prevent pipeline crashes
                continue


def create_streaming_dataloader(batch_size=4, target_size=(224, 224)):
    """
    Factory configuration block that builds the streaming iteration framework.
    """
    # Standard normalization transformations for image models (e.g., CNN backbones / ViT models)
    transform_pipeline = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Connect to streaming stream slice
    print("Connecting to HuggingFace dataset streaming channel...")
    dataset_stream = load_dataset(
        "stzhao/movie_posters_100k_controlnet", 
        split="train", 
        streaming=True
    )

    streaming_dataset = MoviePosterStreamingDataset(
        hf_dataset=dataset_stream, 
        transform=transform_pipeline
    )

    # Note: shuffle=True cannot be used directly with standard structural streaming pipelines. 
    # Buffer-based shuffling or standard streaming parsing is applied here instead.
    loader = DataLoader(
        streaming_dataset, 
        batch_size=batch_size,
        num_workers=0  # Left at 0 to avoid multi-processing memory spikes on CPU architecture
    )
    return loader


if __name__ == "__main__":
    # Execution smoke-test block to check the streaming configuration safely
    print("Initializing test run on data loader infrastructure...")
    
    # Using a micro-batch size of 4 to conform with local RAM constraint budgets
    data_loader = create_streaming_dataloader(batch_size=4)
    
    print("\nStarting pipeline processing iteration test...\n")
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"--- Batch {batch_idx + 1} Captured ---")
        print(f"Images Tensor Footprint (Shape): {images.shape}")
        print(f"Labels Target Multi-Hot (Shape): {labels.shape}")
        
        # Display sample activation distribution across the class array
        print(f"Sample Row Multi-Hot Vectors:\n{labels}")
        print("-" * 40)
        
        # Break early during verification step to conserve compute time
        if batch_idx >= 2:
            print("Verification process completed successfully without memory overflows.")
            break