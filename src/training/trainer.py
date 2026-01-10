"""Training and evaluation functions."""
import os
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    hamming_loss,
    accuracy_score
)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    logging_steps: int = 50
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        logging_steps: Log every N steps

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % logging_steps == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate model on validation/test set.

    Args:
        model: Model to evaluate
        eval_loader: Evaluation dataloader
        device: Device to evaluate on
        threshold: Probability threshold for binary classification

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            total_loss += outputs['loss'].item()
            probs = outputs['probs'].cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)

    # Concatenate all batches
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)

    # Calculate metrics
    avg_loss = total_loss / len(eval_loader)

    # Hamming loss (fraction of wrong labels)
    ham_loss = hamming_loss(all_labels, all_preds)

    # Exact match ratio (all labels must be correct)
    exact_match = accuracy_score(all_labels, all_preds)

    # Per-class metrics (macro and micro averaging)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    # Per-class metrics for each label
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

    metrics = {
        'eval_loss': avg_loss,
        'hamming_loss': ham_loss,
        'exact_match': exact_match,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
    }

    # Add per-class metrics
    label_names = [
        'article_start', 'article_continuation', 'article_end',
        'illustrated_plate', 'plate_caption', 'blank_page', 'other'
    ]
    for i, label_name in enumerate(label_names):
        metrics[f'{label_name}_precision'] = precision_per_class[i]
        metrics[f'{label_name}_recall'] = recall_per_class[i]
        metrics[f'{label_name}_f1'] = f1_per_class[i]
        metrics[f'{label_name}_support'] = support[i]

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: str,
    max_grad_norm: float = 1.0,
    logging_steps: int = 50,
    save_steps: int = 500,
    early_stopping_patience: int = 5,
    early_stopping_metric: str = 'eval_loss',
    save_total_limit: int = 3
) -> Dict[str, list]:
    """Full training loop with checkpointing and early stopping.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        max_grad_norm: Maximum gradient norm
        logging_steps: Log every N steps
        save_steps: Save checkpoint every N steps
        early_stopping_patience: Epochs to wait before early stopping
        early_stopping_metric: Metric to monitor for early stopping
        save_total_limit: Maximum number of checkpoints to keep

    Returns:
        Dictionary with training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'eval_loss': [],
        'f1_macro': [],
        'exact_match': []
    }

    best_metric = float('inf') if 'loss' in early_stopping_metric else 0.0
    patience_counter = 0
    checkpoints = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps
        )

        # Evaluate
        eval_metrics = evaluate_model(
            model=model,
            eval_loader=val_loader,
            device=device
        )

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Log metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"F1 Macro: {eval_metrics['f1_macro']:.4f}")
        print(f"Exact Match: {eval_metrics['exact_match']:.4f}")

        history['train_loss'].append(train_metrics['train_loss'])
        history['eval_loss'].append(eval_metrics['eval_loss'])
        history['f1_macro'].append(eval_metrics['f1_macro'])
        history['exact_match'].append(eval_metrics['exact_match'])

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"checkpoint_epoch_{epoch + 1}.pt"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
        }, checkpoint_path)

        checkpoints.append(checkpoint_path)

        # Remove old checkpoints
        if len(checkpoints) > save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        # Check for improvement
        current_metric = eval_metrics[early_stopping_metric]
        is_better = (
            current_metric < best_metric if 'loss' in early_stopping_metric
            else current_metric > best_metric
        )

        if is_better:
            best_metric = current_metric
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'eval_metrics': eval_metrics,
            }, best_model_path)
            print(f"New best model saved! {early_stopping_metric}: {best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    print("\nTraining completed!")
    return history
