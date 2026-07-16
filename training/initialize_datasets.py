from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataloader import MoviePosterDataset
import torch

def create_dataset():

    # Your existing "baseline" transforms (NO augmentation — already defined as shared_transforms)
    baseline_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = MoviePosterDataset(
        parquet_path="metadata_one_hot_12genres.parquet",
        img_dir="Posters",
        transform=baseline_transforms
    )
    
    NUM_GENRES = len(full_dataset.genre_cols)
    print(f"Dataset successfully compiled. Found {NUM_GENRES} active target classes.")

    total_samples = len(full_dataset)
    train_size = int(0.80 * total_samples) # 80% for training
    val_size   = int(0.10 * total_samples) # 10% for validation checkpoints
    test_size  = total_samples - train_size - val_size # 10% for final test evaluation

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Seeded to guarantee identical evaluation  subsets
    )

    # Initialize Dataloaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    return full_dataset, train_loader, val_loader, test_loader

# =============================================
# DATA AUGMENTATION EXPERIMENT SETUP
# =============================================
def create_augmented_dataset():
    # Your existing "baseline" transforms (NO augmentation — already defined as shared_transforms)
    baseline_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # NEW: Augmented transforms for training data
    augmented_transforms = transforms.Compose([
        transforms.Resize((256, 256)),                          # Resize slightly larger for random crop
        transforms.RandomCrop(224),                             # Random 224x224 crop
        transforms.RandomHorizontalFlip(p=0.5),                 # 50% chance horizontal flip
        transforms.RandomRotation(degrees=15),                  # Rotate ±15°
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variations
        transforms.RandomGrayscale(p=0.1),                      # 10% chance grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a SEPARATE dataset with augmented transforms
    augmented_dataset = MoviePosterDataset(
        parquet_path="metadata_one_hot_12genres.parquet",
        img_dir="Posters",
        transform=augmented_transforms
    )

    # Use the SAME split indices as the baseline to ensure a fair comparison
    # We reuse the exact same train/val/test split via the same seed
    total_samples_aug = len(augmented_dataset)
    train_size_aug = int(0.80 * total_samples_aug)
    val_size_aug   = int(0.10 * total_samples_aug)
    test_size_aug  = total_samples_aug - train_size_aug - val_size_aug

    aug_train_set, aug_val_set, aug_test_set = torch.utils.data.random_split(
        augmented_dataset, [train_size_aug, val_size_aug, test_size_aug],
        generator=torch.Generator().manual_seed(42)  # SAME SEED — critical for fair comparison
    )

    # IMPORTANT: Validation & test sets should always use baseline (no-augmentation) transforms.
    # Since we split first and transform on __getitem__, the val/test subsets will still use
    # augmented transforms. To fix this properly, we override the val/test datasets:

    # Create a separate baseline dataset for val/test (same split, no augmentation)
    baseline_dataset_for_val = MoviePosterDataset(
        parquet_path="metadata_one_hot_12genres.parquet",
        img_dir="Posters",
        transform=baseline_transforms
    )

    _, val_set_baseline, test_set_baseline = torch.utils.data.random_split(
        baseline_dataset_for_val, [train_size_aug, val_size_aug, test_size_aug],
        generator=torch.Generator().manual_seed(42)
    )

    # DataLoaders for the augmented experiment
    aug_train_loader = DataLoader(aug_train_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    # Val and test loaders use the BASELINE transforms (no augmentation during evaluation)
    aug_val_loader   = DataLoader(val_set_baseline, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    aug_test_loader  = DataLoader(test_set_baseline, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    return augmented_dataset, aug_train_loader, aug_val_loader, aug_test_loader
