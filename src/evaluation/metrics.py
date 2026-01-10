"""Evaluation metrics and visualization."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    multilabel_confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    hamming_loss
)
from typing import Dict, List


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_file: str = None
) -> Dict:
    """Calculate comprehensive metrics for multi-label classification.

    Args:
        y_true: True labels [num_samples, num_labels]
        y_pred: Predicted labels [num_samples, num_labels]
        label_names: List of label names
        output_file: Optional path to save report

    Returns:
        Dictionary with metrics
    """
    # Overall metrics
    hamming = hamming_loss(y_true, y_pred)

    # Exact match ratio
    exact_match = np.mean(np.all(y_true == y_pred, axis=1))

    # Per-label metrics
    precision_per_label, recall_per_label, f1_per_label, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    # Averaged metrics
    precision_macro, recall_macro, f1_macro, _ = \
        precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = \
        precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)

    # Create detailed report
    report_data = []
    for i, label in enumerate(label_names):
        report_data.append({
            'Label': label,
            'Precision': precision_per_label[i],
            'Recall': recall_per_label[i],
            'F1-Score': f1_per_label[i],
            'Support': support[i]
        })

    # Add macro and micro averages
    report_data.append({
        'Label': 'Macro Avg',
        'Precision': precision_macro,
        'Recall': recall_macro,
        'F1-Score': f1_macro,
        'Support': np.sum(support)
    })
    report_data.append({
        'Label': 'Micro Avg',
        'Precision': precision_micro,
        'Recall': recall_micro,
        'F1-Score': f1_micro,
        'Support': np.sum(support)
    })

    report_df = pd.DataFrame(report_data)

    # Print report
    print("\n" + "=" * 70)
    print("Classification Report")
    print("=" * 70)
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Exact Match Ratio: {exact_match:.4f}")
    print("\n" + report_df.to_string(index=False))
    print("=" * 70)

    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Hamming Loss: {hamming:.4f}\n")
            f.write(f"Exact Match Ratio: {exact_match:.4f}\n\n")
            f.write(report_df.to_string(index=False))
        print(f"\nReport saved to {output_file}")

    # Return metrics dict
    metrics = {
        'hamming_loss': hamming,
        'exact_match': exact_match,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro
    }

    for i, label in enumerate(label_names):
        metrics[f'{label}_precision'] = precision_per_label[i]
        metrics[f'{label}_recall'] = recall_per_label[i]
        metrics[f'{label}_f1'] = f1_per_label[i]
        metrics[f'{label}_support'] = support[i]

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_file: str = None
):
    """Plot confusion matrices for each label.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names
        output_file: Optional path to save figure
    """
    # Get confusion matrix for each label
    cm = multilabel_confusion_matrix(y_true, y_pred)

    # Create subplots
    n_labels = len(label_names)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_labels > 1 else [axes]

    for i, (label, matrix) in enumerate(zip(label_names, cm)):
        ax = axes[i]

        # Plot heatmap
        sns.heatmap(
            matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=ax,
            cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1']
        )

        ax.set_title(f'{label}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Hide unused subplots
    for i in range(n_labels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {output_file}")

    plt.show()


def plot_label_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    output_file: str = None
):
    """Plot label distribution comparison.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names
        output_file: Optional path to save figure
    """
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)

    x = np.arange(len(label_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)

    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_title('Label Distribution: True vs Predicted')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Label distribution plot saved to {output_file}")

    plt.show()


def analyze_sequential_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filenames: List[str],
    label_names: List[str],
    output_file: str = None
) -> pd.DataFrame:
    """Analyze errors in sequential context.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        filenames: List of filenames
        label_names: List of label names
        output_file: Optional path to save error report

    Returns:
        DataFrame with error analysis
    """
    # Find misclassified samples
    errors = []

    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            true_labels = [label_names[j] for j in range(len(label_names)) if y_true[i, j] == 1]
            pred_labels = [label_names[j] for j in range(len(label_names)) if y_pred[i, j] == 1]

            errors.append({
                'filename': filenames[i],
                'true_labels': ', '.join(true_labels) if true_labels else 'none',
                'predicted_labels': ', '.join(pred_labels) if pred_labels else 'none',
                'error_type': classify_error_type(y_true[i], y_pred[i], label_names)
            })

    error_df = pd.DataFrame(errors)

    if len(errors) > 0:
        print(f"\nTotal errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")

        # Count error types
        error_type_counts = error_df['error_type'].value_counts()
        print("\nError types:")
        print(error_type_counts)

        if output_file:
            error_df.to_csv(output_file, index=False)
            print(f"\nError analysis saved to {output_file}")
    else:
        print("\nNo errors found! Perfect classification!")

    return error_df


def classify_error_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> str:
    """Classify the type of prediction error.

    Args:
        y_true: True labels for one sample
        y_pred: Predicted labels for one sample
        label_names: List of label names

    Returns:
        Error type description
    """
    # Count false positives and false negatives
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    if fp > 0 and fn > 0:
        return "mixed_error"
    elif fp > 0:
        return "false_positive"
    elif fn > 0:
        return "false_negative"
    else:
        return "correct"
