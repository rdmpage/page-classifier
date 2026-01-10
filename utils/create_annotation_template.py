#!/usr/bin/env python3
"""Utility script to create annotation template from image directory."""
import argparse
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
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    print(f"Creating annotation template from: {args.image_dir}")
    print(f"Output file: {args.output}")

    create_annotation_template(
        image_dir=args.image_dir,
        output_file=args.output,
        extract_page_num=not args.no_extract_page_num
    )

    print("\nNext steps:")
    print(f"1. Open {args.output} in a spreadsheet editor or Label Studio")
    print("2. Label each page by setting 1 for applicable categories:")
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
