#!/usr/bin/env python3
"""Main training script for BHL page classifier."""
import argparse
import yaml
import torch
from transformers import ViTImageProcessor
from src.models import BHLPageClassifier
from src.data import create_dataloaders
from src.training import train_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BHL page classifier"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default=None,
        help='Path to annotations CSV (overrides config)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Path to image directory (overrides config)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    annotations_file = args.annotations or config['data']['annotations_file']
    image_dir = args.image_dir or config['data']['raw_dir']

    print("=" * 60)
    print("BHL Page Classifier Training")
    print("=" * 60)
    print(f"Annotations: {annotations_file}")
    print(f"Images: {image_dir}")
    print(f"Model: {config['model']['name']}")
    print(f"Device: {config['device']}")
    print("=" * 60)

    # Set device
    if config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['training']['seed'])

    # Load processor
    print("\nLoading ViT processor...")
    processor = ViTImageProcessor.from_pretrained(config['model']['name'])

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        annotations_file=annotations_file,
        image_dir=image_dir,
        processor=processor,
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['num_workers'],
        seed=config['training']['seed']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nInitializing model...")
    model = BHLPageClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    )
    model = model.to(device)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Create scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps
    )

    # Load optimizer state if resuming
    if args.checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Train
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['training']['num_epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_metric=config['training']['early_stopping_metric'],
        save_total_limit=config['training']['save_total_limit']
    )

    print("\nTraining completed successfully!")
    print(f"Best model saved to: {config['training']['checkpoint_dir']}/best_model.pt")

    # Final evaluation on test set
    from src.training import evaluate_model
    print("\nEvaluating on test set...")
    checkpoint = torch.load(
        f"{config['training']['checkpoint_dir']}/best_model.pt",
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_model(
        model=model,
        eval_loader=test_loader,
        device=device,
        threshold=config['inference']['threshold']
    )

    print("\nTest Set Results:")
    print(f"Loss: {test_metrics['eval_loss']:.4f}")
    print(f"F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"F1 Micro: {test_metrics['f1_micro']:.4f}")
    print(f"Exact Match: {test_metrics['exact_match']:.4f}")
    print(f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")


if __name__ == '__main__':
    main()
