# Quick Start Guide

Get up and running with the BHL Page Classifier in 5 steps.

## Prerequisites

- Python 3.8 or later
- ~500-1500 labeled page images for training
- M1/M2 Mac, GPU, or CPU

## Setup (5 minutes)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data Preparation (varies)

### Option A: You have images but no labels

```bash
# Create annotation template
python utils/create_annotation_template.py \
    --image-dir path/to/images \
    --output data/annotations/labels.csv

# Open data/annotations/labels.csv in Excel/Numbers/LibreOffice
# Set 1 for applicable categories for each page
# Save the file
```

### Option B: You already have labeled data

Ensure your CSV has these columns:
```
filename,page_num,publication_id,article_start,article_continuation,article_end,illustrated_plate,plate_caption,blank_page,other
page001.jpg,1,pub123,1,0,0,0,0,0,0
page002.jpg,2,pub123,0,1,0,0,0,0,0
...
```

## Configuration (2 minutes)

Edit `config.yaml`:

```yaml
data:
  raw_dir: "data/raw"  # Your image directory
  annotations_file: "data/annotations/labels.csv"

training:
  batch_size: 8  # Reduce to 4 if you get memory errors
  num_epochs: 20
```

## Training (30 minutes - 2 hours)

```bash
python train.py
```

Watch for:
- Validation loss going down
- F1 scores improving
- Best model saved to `checkpoints/best_model.pt`

## Inference (seconds to minutes)

```bash
# Create template for new images
python utils/create_annotation_template.py \
    --image-dir path/to/new/images \
    --output new_data.csv

# Run predictions
python predict.py \
    --model checkpoints/best_model.pt \
    --annotations new_data.csv \
    --image-dir path/to/new/images \
    --output predictions.csv

# View results
cat predictions.csv
```

## What's Next?

- **Evaluate performance**: `python evaluate.py --help`
- **Read full docs**: See `README.md`
- **See examples**: Check `notebooks/example_usage.md`
- **Tune performance**: Adjust threshold, batch size, learning rate in `config.yaml`

## Common Issues

**Out of memory**: Reduce `batch_size` in `config.yaml` to 4 or 2

**Poor accuracy**:
- Need more labeled data (aim for 1000+ pages)
- Check label distribution is balanced
- Increase `num_epochs` to 30-50

**Slow training**:
- Check that MPS/CUDA is being used (watch for "Using device: mps/cuda")
- Reduce image resolution if needed

**Import errors**: Make sure you've activated the virtual environment

## Quick Command Reference

```bash
# Create labels template
python utils/create_annotation_template.py --image-dir DIR --output FILE.csv

# Train model
python train.py [--config config.yaml] [--checkpoint CHECKPOINT.pt]

# Predict pages
python predict.py --model MODEL.pt --annotations CSV --image-dir DIR --output OUT.csv

# Evaluate model
python evaluate.py --model MODEL.pt --annotations CSV --image-dir DIR --output-dir DIR

# Resume training
python train.py --checkpoint checkpoints/checkpoint_epoch_5.pt

# Predict without post-processing
python predict.py --model MODEL.pt --no-postprocess ...

# Custom threshold
python predict.py --model MODEL.pt --threshold 0.3 ...
```

## Support

For detailed documentation, see `README.md`
For usage examples, see `notebooks/example_usage.md`
