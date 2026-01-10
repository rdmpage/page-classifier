# Example Usage Guide

This guide shows example workflows for the BHL Page Classifier.

## Example 1: Train from Scratch

```bash
# 1. Create annotation template
python utils/create_annotation_template.py \
    --image-dir data/raw \
    --output data/annotations/labels.csv

# 2. Label your data (manually in spreadsheet or Label Studio)

# 3. Train the model
python train.py --config config.yaml

# 4. Evaluate on test set (uses the test split created during training)
python evaluate.py \
    --model checkpoints/best_model.pt \
    --annotations data/annotations/labels.csv \
    --image-dir data/raw \
    --output-dir evaluation_results \
    --compare-postprocessing
```

## Example 2: Classify New Publication

```bash
# 1. Create template for new images (no labels needed)
python utils/create_annotation_template.py \
    --image-dir /path/to/new/publication \
    --output new_pub_template.csv

# 2. Run predictions
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations new_pub_template.csv \
    --image-dir /path/to/new/publication \
    --output new_pub_predictions.csv

# 3. Review predictions.csv
# Columns: filename, page_num, publication_id,
#          article_start, article_continuation, article_end, etc.
#          article_start_prob, article_continuation_prob, etc.
```

## Example 3: Interactive Analysis (Python)

```python
import pandas as pd
import torch
from transformers import ViTImageProcessor
from src.models import BHLPageClassifier
from src.inference import predict_single_image

# Load model
device = torch.device('mps')  # or 'cuda' or 'cpu'
model = BHLPageClassifier(num_labels=7)
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Load processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# Predict single image
predictions, probs = predict_single_image(
    model=model,
    image_path='path/to/page.jpg',
    processor=processor,
    device=device,
    threshold=0.5
)

# Print results
label_names = [
    'article_start', 'article_continuation', 'article_end',
    'illustrated_plate', 'plate_caption', 'blank_page', 'other'
]

print("Predictions:")
for label, pred, prob in zip(label_names, predictions, probs):
    if pred == 1:
        print(f"  {label}: {prob:.3f}")
```

## Example 4: Batch Processing with Custom Threshold

```bash
# Use lower threshold for recall-focused classification
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations data.csv \
    --image-dir images/ \
    --output predictions_low_threshold.csv \
    --threshold 0.3

# Use higher threshold for precision-focused classification
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations data.csv \
    --image-dir images/ \
    --output predictions_high_threshold.csv \
    --threshold 0.7
```

## Example 5: Analyzing Predictions

```python
import pandas as pd

# Load predictions
df = pd.read_csv('predictions.csv')

# Count pages by type
print("Pages by classification:")
print(df[['article_start', 'article_continuation', 'article_end',
          'illustrated_plate', 'plate_caption', 'blank_page', 'other']].sum())

# Find pages with multiple labels
multi_label = df[df[['article_start', 'article_continuation', 'article_end',
                      'illustrated_plate', 'plate_caption', 'blank_page', 'other']].sum(axis=1) > 1]
print(f"\nPages with multiple labels: {len(multi_label)}")

# Find article boundaries
article_starts = df[df['article_start'] == 1]
print(f"\nNumber of articles: {len(article_starts)}")

# Extract specific publications
pub_id = 'pub123'
pub_pages = df[df['publication_id'] == pub_id].sort_values('page_num')
print(f"\nPages in {pub_id}:")
print(pub_pages[['filename', 'page_num', 'article_start', 'article_end']])
```

## Example 6: Resume Training

```bash
# If training was interrupted, resume from last checkpoint
python train.py \
    --config config.yaml \
    --checkpoint checkpoints/checkpoint_epoch_10.pt
```

## Example 7: Disable Post-Processing

```bash
# Get raw model predictions without sequential corrections
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations data.csv \
    --image-dir images/ \
    --output raw_predictions.csv \
    --no-postprocess
```

## Example 8: Working with Label Distribution

```python
from src.data import BHLPageDataset
from transformers import ViTImageProcessor

# Load dataset
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
dataset = BHLPageDataset(
    annotations_file='data/annotations/labels.csv',
    image_dir='data/raw',
    processor=processor
)

# Check label distribution
distribution = dataset.get_label_distribution()
print("Label distribution:")
print(distribution)

# Get pages from specific publication
pub_ids = dataset.get_publication_ids()
print(f"\nPublications: {pub_ids}")

# Get sequence for one publication
indices = dataset.get_sequence(pub_ids[0])
print(f"\nPages in {pub_ids[0]}: {len(indices)}")
```

## Tips

- Always keep `page_num` and `publication_id` accurate for best sequential post-processing
- Start with a small labeled dataset (100-200 pages) to validate the pipeline
- Use `--compare-postprocessing` flag in evaluation to see the impact of sequential rules
- Monitor per-class F1 scores to identify classes that need more training data
- Adjust threshold based on your use case (recall vs precision trade-off)
