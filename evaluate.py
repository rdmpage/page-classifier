#!/usr/bin/env python3
"""Evaluation script for BHL page classifier."""
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from transformers import ViTImageProcessor
from src.models import BHLPageClassifier
from src.data import create_inference_dataloader
from src.inference import predict_pages
from src.postprocessing import SequentialPostProcessor
from src.evaluation import calculate_metrics, plot_confusion_matrix, plot_label_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate BHL page classifier"
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
        help='Path to test annotations CSV with ground truth labels'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Path to image directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--compare-postprocessing',
        action='store_true',
        help='Compare results with and without post-processing'
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("BHL Page Classifier - Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Test data: {args.annotations}")
    print(f"Images: {args.image_dir}")
    print("=" * 70)

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Load ground truth
    print("Loading ground truth labels...")
    gt_df = pd.read_csv(args.annotations)
    label_names = config['labels']
    y_true = gt_df[label_names].values

    # Load model
    print("\nLoading model...")
    model = BHLPageClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Run inference
    print("\nRunning inference...")
    predictions, probabilities, metadata = predict_pages(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=config['inference']['threshold'],
        return_probs=True
    )

    # Evaluate without post-processing
    print("\n" + "=" * 70)
    print("EVALUATION WITHOUT POST-PROCESSING")
    print("=" * 70)

    metrics_raw = calculate_metrics(
        y_true=y_true,
        y_pred=predictions,
        label_names=label_names,
        output_file=f"{args.output_dir}/report_raw.txt"
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=predictions,
        label_names=label_names,
        output_file=f"{args.output_dir}/confusion_matrix_raw.png"
    )

    # Plot label distribution
    plot_label_distribution(
        y_true=y_true,
        y_pred=predictions,
        label_names=label_names,
        output_file=f"{args.output_dir}/label_distribution_raw.png"
    )

    # Evaluate with post-processing if enabled
    if args.compare_postprocessing or config['sequential']['enabled']:
        print("\n" + "=" * 70)
        print("EVALUATION WITH POST-PROCESSING")
        print("=" * 70)

        post_processor = SequentialPostProcessor(
            min_article_length=config['sequential']['min_article_length'],
            isolated_correction=config['sequential']['isolated_correction'],
            blank_tolerance=config['sequential']['blank_tolerance'],
            label_names=label_names
        )

        # Apply post-processing
        if metadata['publication_ids'] and metadata['publication_ids'][0] != 'unknown':
            predictions_pp, stats = post_processor.process_by_publication(
                predictions=predictions.copy(),
                publication_ids=metadata['publication_ids'],
                filenames=metadata['filenames'],
                page_nums=metadata['page_nums']
            )
        else:
            predictions_pp, stats = post_processor.process_sequence(
                predictions=predictions.copy(),
                filenames=metadata['filenames'],
                page_nums=metadata['page_nums']
            )

        print(f"\nPost-processing statistics:")
        print(f"  Corrections made: {stats['corrections_made']}")
        print(f"  Isolated articles fixed: {stats['isolated_articles_fixed']}")
        print(f"  Boundary corrections: {stats['boundary_corrections']}")
        print(f"  Continuation insertions: {stats['continuation_insertions']}")

        # Evaluate post-processed predictions
        metrics_pp = calculate_metrics(
            y_true=y_true,
            y_pred=predictions_pp,
            label_names=label_names,
            output_file=f"{args.output_dir}/report_postprocessed.txt"
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=predictions_pp,
            label_names=label_names,
            output_file=f"{args.output_dir}/confusion_matrix_postprocessed.png"
        )

        # Plot label distribution
        plot_label_distribution(
            y_true=y_true,
            y_pred=predictions_pp,
            label_names=label_names,
            output_file=f"{args.output_dir}/label_distribution_postprocessed.png"
        )

        # Compare metrics
        print("\n" + "=" * 70)
        print("COMPARISON: Raw vs Post-Processed")
        print("=" * 70)
        comparison = pd.DataFrame({
            'Metric': ['F1 Macro', 'F1 Micro', 'Exact Match', 'Hamming Loss'],
            'Raw': [
                metrics_raw['f1_macro'],
                metrics_raw['f1_micro'],
                metrics_raw['exact_match'],
                metrics_raw['hamming_loss']
            ],
            'Post-Processed': [
                metrics_pp['f1_macro'],
                metrics_pp['f1_micro'],
                metrics_pp['exact_match'],
                metrics_pp['hamming_loss']
            ],
            'Improvement': [
                metrics_pp['f1_macro'] - metrics_raw['f1_macro'],
                metrics_pp['f1_micro'] - metrics_raw['f1_micro'],
                metrics_pp['exact_match'] - metrics_raw['exact_match'],
                metrics_raw['hamming_loss'] - metrics_pp['hamming_loss']  # Lower is better
            ]
        })
        print(comparison.to_string(index=False))
        comparison.to_csv(f"{args.output_dir}/comparison.csv", index=False)

    print("\n" + "=" * 70)
    print(f"Evaluation complete! Results saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
