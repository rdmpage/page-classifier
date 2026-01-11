# Folder-Based Labeling Workflow

This guide explains how to use a folder-based approach for labeling, where you organize images into folders by category and then generate a CSV file.

## Your Workflow

1. Organize images into folders by label
2. Generate CSV using your PHP script
3. Update config.yaml with your labels
4. Train the model

## How Labels Work (Now Fully Dynamic!)

The system automatically detects labels from your CSV file. Any column that isn't `filename`, `page_num`, or `publication_id` is treated as a label.

### What You Need to Do:

#### 1. Create Your CSV (via PHP or any method)

Your CSV should have this structure:

```csv
filename,page_num,publication_id,cover,article_start,article_continuation,article_end,illustrated_plate,blank_page,other
page001.jpg,1,pub123,1,0,0,0,0,0,0
page002.jpg,2,pub123,0,1,0,0,0,0,0
page003.jpg,3,pub123,0,0,1,0,0,0,0
page004.jpg,4,pub123,0,0,0,1,0,0,0
...
```

**Rules:**
- First 3 columns are metadata: `filename`, `page_num`, `publication_id`
- All other columns are treated as labels
- Label values are 0 or 1
- You can have any number of labels
- Pages can have multiple labels (e.g., `cover=1, article_start=1`)

#### 2. Update config.yaml

List your labels in the config (this is used for display and post-processing logic):

```yaml
# config.yaml
model:
  num_labels: 7  # Update this to match your number of labels

labels:
  - "cover"                    # Your new label
  - "article_start"
  - "article_continuation"
  - "article_end"
  - "illustrated_plate"
  - "blank_page"
  - "other"
```

#### 3. Train

```bash
python train.py --config config.yaml
```

The system will automatically use all labels from your CSV!

## Example PHP Script Pattern

Here's the general approach your PHP script should follow:

```php
<?php
// Pseudo-code for generating CSV from folders

$folders = [
    'cover' => '/path/to/cover/images',
    'article_start' => '/path/to/article_starts',
    'illustrated_plate' => '/path/to/plates',
    // ... etc
];

$labels = array_keys($folders);
$rows = [];

foreach ($folders as $label => $folder) {
    foreach (scandir($folder) as $file) {
        if (!is_image($file)) continue;

        // Initialize row
        $row = [
            'filename' => $file,
            'page_num' => extract_page_num($file),
            'publication_id' => extract_pub_id($file)
        ];

        // Set all labels to 0
        foreach ($labels as $l) {
            $row[$l] = 0;
        }

        // Set this label to 1
        $row[$label] = 1;

        $rows[] = $row;
    }
}

// Write CSV
write_csv('data/annotations/labels.csv', $rows);
?>
```

## Multi-Label Pages

If a page belongs to multiple categories, you can handle it two ways:

### Option 1: Duplicate in multiple folders
Put the same image in multiple folders. Your script should merge them:

```php
// If page001.jpg appears in both 'cover' and 'article_start':
$row['cover'] = 1;
$row['article_start'] = 1;
```

### Option 2: Special multi-label folder
Create a folder for special cases with a manifest file:

```
multi-label/
  page001.jpg  -> cover=1, article_start=1
  page002.jpg  -> article_end=1, illustrated_plate=1
```

## Important Notes

### ✅ What's Automatic:
- Label detection from CSV columns
- Number of labels (counts non-metadata columns)
- Dataset loading and validation

### ⚠️ What You Must Update:

1. **config.yaml**:
   - `model.num_labels` - must match your label count
   - `labels` list - for display purposes and evaluation

2. **Your CSV**:
   - Must have `filename`, `page_num`, `publication_id` as first 3 columns
   - All other columns are labels (0/1 values)

### Sequential Post-Processing

If you use labels like `cover`, `table_of_contents`, etc., they won't be affected by the sequential post-processing (which only looks for `article_start`, `article_continuation`, `article_end`, and `blank_page`).

If you want to add custom sequential logic for your new labels, you'd need to modify `src/postprocessing/sequential.py`.

## Example Label Sets

### Minimal (3 labels):
```yaml
labels:
  - "text"
  - "illustration"
  - "blank"
```

### Extended (10 labels):
```yaml
labels:
  - "cover"
  - "title_page"
  - "table_of_contents"
  - "article_start"
  - "article_continuation"
  - "article_end"
  - "illustrated_plate"
  - "plate_caption"
  - "blank_page"
  - "other"
```

### Domain-Specific (biology journals):
```yaml
labels:
  - "taxonomy"
  - "species_description"
  - "dichotomous_key"
  - "distribution_map"
  - "specimen_illustration"
  - "bibliography"
  - "blank"
```

## Testing Your Setup

1. Create a small test CSV with your labels:
```bash
# Test with 10 images
head -11 data/annotations/labels.csv
```

2. Verify label detection:
```python
from src.data import BHLPageDataset
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
dataset = BHLPageDataset('data/annotations/labels.csv', 'data/raw', processor)

print(f"Detected labels: {dataset.label_columns}")
print(f"Number of labels: {len(dataset.label_columns)}")
print(f"Label distribution:\n{dataset.get_label_distribution()}")
```

3. Update `config.yaml` to match the number of labels

4. Run a quick training test:
```bash
# Reduce epochs for quick test
python train.py --config config.yaml
```

## Troubleshooting

**"No label columns found!"**
- Make sure your CSV has columns other than `filename`, `page_num`, `publication_id`

**"Model num_labels doesn't match data"**
- Count your label columns (excluding metadata)
- Update `model.num_labels` in config.yaml to match

**"Label X not in config"**
- The code will still work, but add it to `labels` list in config.yaml for better reporting

**Poor performance on new labels**
- Collect at least 50-100 examples per label
- Check class balance: `dataset.get_label_distribution()`

---

**Summary:** Just create your CSV with any labels you want, update the count in config.yaml, and the system handles the rest automatically!
