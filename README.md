# BHL Page Classification Project

A machine learning system for classifying page images from the Biodiversity Heritage Library (BHL) and other historical document collections into semantic categories.

## Overview

This project uses a fine-tuned Vision Transformer (ViT) model to classify pages into multiple categories:

- **Article start page**: First page of an article
- **Article continuation page**: Middle pages of an article
- **Article end page**: Last page of an article
- **Illustrated plate**: Pages with illustrations/figures
- **Plate caption page**: Caption pages for plates
- **Blank page**: Blank or nearly blank pages
- **Other/irrelevant**: Other types of content

### Key Features

- **Multi-label classification**: Pages can belong to multiple categories simultaneously
- **Sequential post-processing**: Uses page order to refine predictions and fix inconsistencies
- **M1 Mac optimized**: Supports Apple Silicon MPS acceleration
- **Comprehensive evaluation**: Detailed metrics and visualizations

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── train.py                    # Training script
├── predict.py                  # Inference script
├── evaluate.py                 # Evaluation script
├── data/
│   ├── raw/                    # Raw page images
│   ├── processed/              # Processed data
│   └── annotations/            # Label CSV files
├── src/
│   ├── data/                   # Dataset and dataloader
│   ├── models/                 # Model definitions
│   ├── training/               # Training utilities
│   ├── inference/              # Inference utilities
│   ├── postprocessing/         # Sequential post-processing
│   └── evaluation/             # Evaluation metrics
├── models/                     # Saved models
├── checkpoints/                # Training checkpoints
├── logs/                       # Training logs
└── utils/                      # Utility scripts
```

## Installation

1. **Clone the repository** (or navigate to this directory)

2. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

#### Option A: Create annotation template from images

```bash
python utils/create_annotation_template.py \
    --image-dir path/to/your/images \
    --output data/annotations/labels.csv
```

This creates a CSV template with columns for all labels. Open it in a spreadsheet editor and label your images.

#### Option B: Use existing annotations

Ensure your CSV has these columns:
- `filename`: Image filename
- `page_num`: Page number (for sequential ordering)
- `publication_id`: Publication identifier (optional but recommended)
- Label columns: `article_start`, `article_continuation`, `article_end`, `illustrated_plate`, `plate_caption`, `blank_page`, `other`

Each label column should contain 0 or 1.

### 2. Update Configuration

Edit `config.yaml` to match your setup:

```yaml
data:
  raw_dir: "path/to/your/images"
  annotations_file: "data/annotations/labels.csv"

training:
  batch_size: 8  # Adjust based on your GPU/MPS memory
  num_epochs: 20
```

### 3. Train the Model

```bash
python train.py --config config.yaml
```

This will:
- Load and split your data (70% train, 15% val, 15% test by default)
- Fine-tune the ViT model
- Save checkpoints during training
- Save the best model based on validation loss
- Evaluate on the test set

Training progress and metrics will be displayed in the console.

### 4. Run Inference

Classify new images:

```bash
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations path/to/new_images/template.csv \
    --image-dir path/to/new_images \
    --output predictions.csv
```

The output CSV will contain:
- Original metadata (filename, page_num, publication_id)
- Binary predictions for each label
- Probability scores for each label

### 5. Evaluate Model

Evaluate on a test set with ground truth labels:

```bash
python evaluate.py \
    --model checkpoints/best_model.pt \
    --annotations data/annotations/test_labels.csv \
    --image-dir data/raw \
    --output-dir evaluation_results \
    --compare-postprocessing
```

This generates:
- Classification reports (precision, recall, F1)
- Confusion matrices for each label
- Label distribution plots
- Comparison of raw vs post-processed predictions

## Advanced Usage

### Sequential Post-Processing

The system includes intelligent post-processing that uses page order to correct predictions:

1. **Fix isolated articles**: Single-page articles surrounded by multi-page articles are suspicious
2. **Ensure boundary consistency**: Article start/end markers must be consistent
3. **Fill gaps**: Pages between article start and end are likely continuations
4. **Handle blank pages**: Blank pages within articles are treated specially

Configure in `config.yaml`:

```yaml
sequential:
  enabled: true
  min_article_length: 2
  isolated_correction: true
  blank_tolerance: 1
```

Disable post-processing during inference:

```bash
python predict.py --model checkpoints/best_model.pt --no-postprocess ...
```

### Resume Training from Checkpoint

```bash
python train.py --checkpoint checkpoints/checkpoint_epoch_5.pt
```

### Custom Data Splits

Modify `config.yaml`:

```yaml
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

### Adjust Classification Threshold

Default threshold is 0.5. Adjust for your use case:

```bash
python predict.py --threshold 0.3 ...  # More sensitive (more predictions)
python predict.py --threshold 0.7 ...  # More conservative (fewer predictions)
```

### Hardware Configuration

The system automatically detects and uses available hardware. Configure in `config.yaml`:

```yaml
device: "mps"      # For M1/M2 Macs
# device: "cuda"   # For NVIDIA GPUs
# device: "cpu"    # For CPU only
```

## Data Labeling Tips

### Recommended Workflow

1. **Sample diverse publications**: Choose 10-20 different publications
2. **Label sequences**: Label 20-30 consecutive pages from each publication
3. **Maintain order**: Keep page numbers and publication IDs accurate
4. **Multi-label examples**:
   - A page where one article ends and another starts: `article_end=1, article_start=1`
   - An illustrated plate with caption: `illustrated_plate=1, plate_caption=1`

### Label Definitions

- **article_start**: First page of a scientific article or chapter (often has title, author)
- **article_continuation**: Body pages of an article
- **article_end**: Last page of an article (may have references, acknowledgments)
- **illustrated_plate**: Full page or partial page illustration/figure
- **plate_caption**: Page or section describing/captioning illustrations
- **blank_page**: Completely blank or nearly blank pages
- **other**: Table of contents, indices, advertisements, etc.

### Pre-labeling Heuristics

You can use these rules to pre-label, then manually verify:

- Nearly all white pixels → likely `blank_page`
- Very little text, mostly images → likely `illustrated_plate`
- Contains "Fig.", "Plate", "Figure" text → likely `plate_caption`

## Model Architecture

- **Base model**: `google/vit-base-patch16-224`
- **Input**: 224×224 RGB images (automatically resized)
- **Output**: 7 class probabilities (multi-label via sigmoid)
- **Parameters**: ~86M total (base ViT + classification head)

### Training Details

- **Optimizer**: AdamW with weight decay
- **Learning rate**: 2e-5 with linear warmup and decay
- **Loss**: Binary cross-entropy with logits (BCEWithLogitsLoss)
- **Early stopping**: Based on validation loss
- **Checkpointing**: Saves best model and recent checkpoints

## Evaluation Metrics

The system reports comprehensive metrics:

- **Hamming Loss**: Fraction of incorrectly predicted labels
- **Exact Match**: Percentage of samples with all labels correct
- **F1 Macro**: Average F1 across all classes (unweighted)
- **F1 Micro**: Global F1 considering all predictions
- **Per-class metrics**: Precision, recall, F1 for each label

## Troubleshooting

### Out of Memory Errors

Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 4  # Reduce from 8
```

### Poor Performance on Specific Classes

1. Check class balance: Use `dataset.get_label_distribution()`
2. Collect more examples of underrepresented classes
3. Adjust classification threshold per class if needed

### Sequential Post-Processing Making Wrong Corrections

Disable specific rules in `config.yaml`:

```yaml
sequential:
  isolated_correction: false  # Keep single-page articles
  blank_tolerance: 0          # Don't allow blanks within articles
```

## Future Enhancements

Potential Stage 3 improvements:

- **Sliding window context**: Feed 3-5 consecutive pages to the model
- **Sequence-to-sequence model**: Use LSTM/Transformer to model page sequences
- **Active learning**: Identify uncertain predictions for manual labeling
- **Data augmentation**: Rotation, brightness, contrast adjustments

## Citation

If you use this code, please cite:

```
BHL Page Classification System
https://github.com/yourusername/page-classifier
```

## License

[Specify your license here]

## Contact

[Your contact information]

---

**Note**: This system requires labeled training data. Start with 500-1500 labeled pages for initial experiments. More data generally improves performance.