#!/usr/bin/env python3
"""Utility script to create annotation template from image directory."""
import argparse
import sys
from pathlib import Path
import yaml

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_annotation_template


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create annotation template CSV from images"
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Directory containing page images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/annotations/labels.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--no-extract-page-num',
        action='store_true',
        help='Do not try to extract page numbers from filenames'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file with label definitions'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load labels from config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        label_columns = config.get('labels', None)
    else:
        print(f"Warning: Config file {config_path} not found, using default labels")
        label_columns = None

    print(f"Creating annotation template from: {args.image_dir}")
    print(f"Output file: {args.output}")
    if label_columns:
        print(f"Using labels from config: {', '.join(label_columns)}")

    create_annotation_template(
        image_dir=args.image_dir,
        output_file=args.output,
        extract_page_num=not args.no_extract_page_num,
        label_columns=label_columns
    )

    print("\nNext steps:")
    print(f"1. Open {args.output} in a spreadsheet editor or Label Studio")
    print("2. Label each page by setting 1 for applicable categories:")
    if label_columns:
        for label in label_columns:
            print(f"   - {label}")
    else:
        print("   - article_start: First page of an article")
        print("   - article_continuation: Middle pages of an article")
        print("   - article_end: Last page of an article")
        print("   - illustrated_plate: Page with illustration/figure")
        print("   - plate_caption: Caption page for plates")
        print("   - blank_page: Blank or nearly blank page")
        print("   - other: Other types of content")
    print("3. Note: Pages can have multiple labels (multi-label classification)")
    print("4. Save the CSV file and use it for training")


if __name__ == '__main__':
    main()
