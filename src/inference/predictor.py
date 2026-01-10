"""Inference functions for page classification."""
import torch
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm


def predict_pages(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    return_probs: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Run inference on pages.

    Args:
        model: Trained model
        dataloader: DataLoader with images
        device: Device to run on
        threshold: Probability threshold for binary classification
        return_probs: Whether to return probabilities

    Returns:
        Tuple of (predictions, probabilities, metadata)
        - predictions: [num_pages, num_labels] binary array
        - probabilities: [num_pages, num_labels] probability array
        - metadata: dict with filenames, page_nums, publication_ids
    """
    model.eval()

    all_probs = []
    all_filenames = []
    all_page_nums = []
    all_pub_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(device)

            outputs = model(pixel_values=pixel_values)
            probs = outputs['probs'].cpu().numpy()

            all_probs.append(probs)
            all_filenames.extend(batch['filenames'])
            all_page_nums.extend(batch['page_nums'])
            all_pub_ids.extend(batch['publication_ids'])

    # Concatenate all batches
    all_probs = np.vstack(all_probs)
    predictions = (all_probs >= threshold).astype(int)

    metadata = {
        'filenames': all_filenames,
        'page_nums': all_page_nums,
        'publication_ids': all_pub_ids
    }

    if return_probs:
        return predictions, all_probs, metadata
    else:
        return predictions, None, metadata


def predict_single_image(
    model: torch.nn.Module,
    image_path: str,
    processor,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict labels for a single image.

    Args:
        model: Trained model
        image_path: Path to image
        processor: ViT image processor
        device: Device to run on
        threshold: Probability threshold

    Returns:
        Tuple of (predictions, probabilities)
    """
    from PIL import Image

    model.eval()

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(image, return_tensors="pt")['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = outputs['probs'].cpu().numpy()[0]

    predictions = (probs >= threshold).astype(int)

    return predictions, probs
