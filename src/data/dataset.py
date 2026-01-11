"""Dataset class for BHL page classification with sequence preservation."""
import os
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor


class BHLPageDataset(Dataset):
    """Dataset for BHL page images with multi-label classification.

    Expected CSV format:
        filename,page_num,publication_id,article_start,article_continuation,
        article_end,illustrated_plate,plate_caption,blank_page,other

    Where label columns contain 0/1 for each class.
    Page order is preserved through page_num column.
    """

    def __init__(
        self,
        annotations_file: str,
        image_dir: str,
        processor: ViTImageProcessor,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            annotations_file: Path to CSV with labels
            image_dir: Directory containing images
            processor: HuggingFace ViT image processor
            transform: Optional torchvision transforms
        """
        self.annotations = pd.read_csv(annotations_file)
        self.image_dir = image_dir
        self.processor = processor
        self.transform = transform

        # Sort by publication_id and page_num to maintain sequence
        if 'publication_id' in self.annotations.columns:
            self.annotations = self.annotations.sort_values(
                ['publication_id', 'page_num']
            ).reset_index(drop=True)
        elif 'page_num' in self.annotations.columns:
            self.annotations = self.annotations.sort_values('page_num').reset_index(drop=True)

        # Auto-detect label columns: any column that's not metadata
        metadata_cols = {'filename', 'page_num', 'publication_id'}
        self.label_columns = [
            col for col in self.annotations.columns
            if col not in metadata_cols
        ]

        if len(self.label_columns) == 0:
            raise ValueError(
                f"No label columns found! CSV must have columns other than "
                f"{metadata_cols}. Found columns: {list(self.annotations.columns)}"
            )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index.

        Returns:
            Dictionary with 'pixel_values', 'labels', and metadata
        """
        row = self.annotations.iloc[idx]

        # Load image
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        # Apply custom transforms if provided
        if self.transform:
            image = self.transform(image)

        # Process with ViT processor
        pixel_values = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)

        # Extract multi-label targets
        labels = torch.tensor(
            row[self.label_columns].values.astype(float),
            dtype=torch.float32
        )

        # Return with metadata for sequential processing
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'filename': row['filename'],
            'page_num': row.get('page_num', idx),
            'publication_id': row.get('publication_id', 'unknown'),
            'idx': idx
        }

    def get_sequence(self, publication_id: str) -> List[int]:
        """Get indices for all pages in a publication sequence.

        Args:
            publication_id: Publication identifier

        Returns:
            List of indices for pages in this publication
        """
        if 'publication_id' not in self.annotations.columns:
            raise ValueError("Dataset does not have publication_id column")

        mask = self.annotations['publication_id'] == publication_id
        return self.annotations[mask].index.tolist()

    def get_label_distribution(self) -> pd.Series:
        """Get distribution of labels in dataset.

        Returns:
            Series with count of positive labels for each class
        """
        return self.annotations[self.label_columns].sum()

    def get_publication_ids(self) -> List[str]:
        """Get list of unique publication IDs in dataset.

        Returns:
            List of publication IDs
        """
        if 'publication_id' not in self.annotations.columns:
            return []
        return self.annotations['publication_id'].unique().tolist()


def create_annotation_template(
    image_dir: str,
    output_file: str,
    extract_page_num: bool = True,
    label_columns: List[str] = None
) -> None:
    """Create a template CSV for annotation.

    Args:
        image_dir: Directory containing images
        output_file: Path to save template CSV
        extract_page_num: Try to extract page numbers from filenames
        label_columns: List of label column names (if None, uses defaults)
    """
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    ]

    data = {'filename': image_files}

    # Try to extract page numbers from filename patterns like:
    # "pub123_page045.jpg" or "item_0045.jpg"
    if extract_page_num:
        page_nums = []
        pub_ids = []

        for filename in image_files:
            # Try to find page number
            page_match = re.search(r'(?:page|p)?[_-]?(\d+)', filename, re.IGNORECASE)
            page_nums.append(int(page_match.group(1)) if page_match else 0)

            # Try to find publication ID
            pub_match = re.search(r'(pub|item|book)?[_-]?(\w+)(?:_page|_p)?', filename, re.IGNORECASE)
            pub_ids.append(pub_match.group(2) if pub_match else 'unknown')

        data['page_num'] = page_nums
        data['publication_id'] = pub_ids

    # Add empty label columns
    if label_columns is None:
        label_columns = [
            'article_start',
            'article_continuation',
            'article_end',
            'illustrated_plate',
            'plate_caption',
            'blank_page',
            'other'
        ]

    for col in label_columns:
        data[col] = 0

    df = pd.DataFrame(data)
    df = df.sort_values(['publication_id', 'page_num'] if extract_page_num else ['filename'])
    df.to_csv(output_file, index=False)
    print(f"Created annotation template at {output_file}")
    print(f"Total images: {len(image_files)}")
