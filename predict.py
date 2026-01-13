#!/usr/bin/env python3
"""Inference script for BHL page classification."""
import argparse
import yaml
import torch
import pandas as pd
from transformers import ViTImageProcessor
from src.models import BHLPageClassifier
from src.data import create_inference_dataloader
from src.inference import predict_pages
from src.postprocessing import SequentialPostProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict page classifications"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        required=True,
        help='Path to annotations CSV (can be template with dummy labels)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Path to image directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Output CSV file for predictions'
    )
    parser.add_argument(
        '--no-postprocess',
        action='store_true',
        help='Disable sequential post-processing'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Probability threshold (overrides config)'
    )
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    threshold = args.threshold or config['inference']['threshold']

    print("=" * 60)
    print("BHL Page Classifier - Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Images: {args.image_dir}")
    print(f"Threshold: {threshold}")
    print(f"Post-processing: {not args.no_postprocess}")
    print("=" * 60)

    # Set device
    if config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load processor
    print("\nLoading ViT processor...")
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])

    # Create dataloader
    print("Creating dataloader...")
    dataloader = create_inference_dataloader(
        annotations_file=args.annotations,
        image_dir=args.image_dir,
        processor=processor,
        batch_size=config['inference']['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"Total batches: {len(dataloader)}")

    # Load model
    print("\nLoading model...")
    model = BHLPageClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded successfully!")

    # Run inference
    print("\nRunning inference...")
    predictions, probabilities, metadata = predict_pages(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=threshold,
        return_probs=True
    )

    print(f"Predicted {len(predictions)} pages")

    # Apply sequential post-processing if enabled
    if not args.no_postprocess and config['sequential']['enabled']:
        print("\nApplying sequential post-processing...")

        post_processor = SequentialPostProcessor(
            min_article_length=config['sequential']['min_article_length'],
            isolated_correction=config['sequential']['isolated_correction'],
            blank_tolerance=config['sequential']['blank_tolerance'],
            label_names=config['labels']
        )

        # Process by publication if available
        if metadata['publication_ids'] and metadata['publication_ids'][0] != 'unknown':
            predictions, stats = post_processor.process_by_publication(
                predictions=predictions,
                publication_ids=metadata['publication_ids'],
                filenames=metadata['filenames'],
                page_nums=metadata['page_nums']
            )
            print(f"Processed {stats['total_publications']} publications")
        else:
            predictions, stats = post_processor.process_sequence(
                predictions=predictions,
                filenames=metadata['filenames'],
                page_nums=metadata['page_nums']
            )

        print(f"Corrections made: {stats['corrections_made']}")
        print(f"  - Isolated articles fixed: {stats['isolated_articles_fixed']}")
        print(f"  - Boundary corrections: {stats['boundary_corrections']}")
        print(f"  - Continuation insertions: {stats['continuation_insertions']}")

    # Save predictions
    print(f"\nSaving predictions to {args.output}...")

    # Create results DataFrame
    results = pd.DataFrame({
        'filename': metadata['filenames'],
        'page_num': metadata['page_nums'],
        'publication_id': metadata['publication_ids']
    })

    # Add predictions and probabilities for each label
    label_names = config['labels']
    for i, label in enumerate(label_names):
        results[label] = predictions[:, i]
        results[f'{label}_prob'] = probabilities[:, i]

    results.to_csv(args.output, index=False)
    print("Done!")

    # Print summary
    print("\nPrediction Summary:")
    for i, label in enumerate(label_names):
        count = predictions[:, i].sum()
        pct = (count / len(predictions)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
