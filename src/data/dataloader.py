"""DataLoader utilities for training and evaluation."""
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from transformers import ViTImageProcessor
from .dataset import BHLPageDataset


def collate_fn(batch):
    """Custom collate function to handle metadata.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched tensors and metadata lists
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'filenames': [item['filename'] for item in batch],
        'page_nums': [item['page_num'] for item in batch],
        'publication_ids': [item['publication_id'] for item in batch],
        'indices': [item['idx'] for item in batch]
    }


def create_dataloaders(
    annotations_file: str,
    image_dir: str,
    processor: ViTImageProcessor,
    batch_size: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    train_transform=None,
    eval_transform=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        annotations_file: Path to annotations CSV
        image_dir: Directory containing images
        processor: ViT image processor
        batch_size: Batch size for training
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        train_transform: Optional transforms for training
        eval_transform: Optional transforms for evaluation

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Split proportions must sum to 1.0"

    # Create full dataset
    full_dataset = BHLPageDataset(
        annotations_file=annotations_file,
        image_dir=image_dir,
        processor=processor,
        transform=train_transform
    )

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Update transform for validation and test if provided
    if eval_transform is not None:
        val_dataset.dataset.transform = eval_transform
        test_dataset.dataset.transform = eval_transform

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def create_inference_dataloader(
    annotations_file: str,
    image_dir: str,
    processor: ViTImageProcessor,
    batch_size: int = 16,
    num_workers: int = 4,
    transform=None
) -> DataLoader:
    """Create dataloader for inference on new images.

    Args:
        annotations_file: Path to annotations CSV
        image_dir: Directory containing images
        processor: ViT image processor
        batch_size: Batch size for inference
        num_workers: Number of workers
        transform: Optional transforms

    Returns:
        DataLoader for inference
    """
    dataset = BHLPageDataset(
        annotations_file=annotations_file,
        image_dir=image_dir,
        processor=processor,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: maintain order for sequential processing
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return loader
