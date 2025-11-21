# Evaluation Guide

Panduan lengkap untuk evaluasi model restorasi 3D.

## 📊 Available Metrics

### 2D Restoration Metrics

#### 1. **PSNR (Peak Signal-to-Noise Ratio)**
- **Range:** 0 - ∞ (higher is better)
- **Good Value:** >28 dB
- **Best for:** Overall image quality
- **Formula:** `20 * log10(MAX / sqrt(MSE))`

#### 2. **SSIM (Structural Similarity Index)**
- **Range:** 0 - 1 (higher is better)
- **Good Value:** >0.85
- **Best for:** Structural similarity
- **Measures:** Luminance, contrast, structure

#### 3. **MAE (Mean Absolute Error)**
- **Range:** 0 - 1 (lower is better)
- **Good Value:** <0.05
- **Best for:** Average pixel difference
- **Formula:** `mean(|predicted - target|)`

#### 4. **MSE (Mean Squared Error)**
- **Range:** 0 - 1 (lower is better)
- **Good Value:** <0.01
- **Best for:** Penalized large errors
- **Formula:** `mean((predicted - target)²)`

#### 5. **LPIPS (Learned Perceptual Image Patch Similarity)**
- **Range:** 0 - 1 (lower is better)
- **Good Value:** <0.15
- **Best for:** Perceptual similarity
- **Requires:** `pip install lpips`

---

## 🚀 Running Evaluation

### Basic Usage

```bash
# Evaluate on test set
python main.py --mode evaluate --test_dir data/test

# With all metrics including LPIPS
python main.py --mode evaluate \
    --test_dir data/test \
    --use_lpips \
    --batch_size 4
```

### Advanced Usage

```bash
# Custom output directory
python main.py --mode evaluate \
    --test_dir outputs/venus_statue \
    --output_dir results/eval_venus \
    --use_lpips

# Using standalone script
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --test_dir data/test \
    --use_lpips
```

---

## 📁 Test Data Structure

Your test directory should have this structure:

```
data/test/
├── images/                  # Clean images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── dense/
    └── depth_maps/         # Depth maps
        ├── image_001.png
        ├── image_002.png
        └── ...
```

**Note:** The evaluation will automatically apply damage simulation to clean images for testing.

---

## 📊 Output Files

After evaluation, you'll find:

```
results/eval/
├── evaluation_results.json      # All metrics in JSON format
├── results_table.md            # Markdown table
└── examples/                   # Visual comparisons (first 5 samples)
    ├── example_0.png
    ├── example_1.png
    └── ...
```

### evaluation_results.json

```json
{
  "psnr": {
    "mean": 28.4523,
    "std": 2.1234,
    "min": 24.5678,
    "max": 34.2109,
    "median": 28.1234
  },
  "ssim": {
    "mean": 0.8912,
    "std": 0.0456,
    "min": 0.7845,
    "max": 0.9534,
    "median": 0.8923
  },
  ...
}
```

### results_table.md

```markdown
# Evaluation Results

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| PSNR | 28.4523 | 2.1234 | 24.5678 | 34.2109 | 28.1234 |
| SSIM | 0.8912 | 0.0456 | 0.7845 | 0.9534 | 0.8923 |
...
```

---

## 📈 Interpreting Results

### Excellent Model
```
PSNR:  >30 dB
SSIM:  >0.90
MAE:   <0.03
LPIPS: <0.10
```

### Good Model
```
PSNR:  28-30 dB
SSIM:  0.85-0.90
MAE:   0.03-0.05
LPIPS: 0.10-0.15
```

### Acceptable Model
```
PSNR:  25-28 dB
SSIM:  0.80-0.85
MAE:   0.05-0.08
LPIPS: 0.15-0.20
```

### Needs Improvement
```
PSNR:  <25 dB
SSIM:  <0.80
MAE:   >0.08
LPIPS: >0.20
```

---

## 🔍 Troubleshooting

### Low PSNR (<25 dB)

**Possible Causes:**
- Insufficient training epochs
- Model not converged
- Learning rate too high/low
- Bad training data

**Solutions:**
```bash
# Train longer
python main.py --mode train --epochs 100

# With pretrained encoder
python main.py --mode train --use_pretrained_encoder --epochs 50
```

### Low SSIM (<0.80)

**Possible Causes:**
- Structural information lost
- Need more perceptual loss
- Augmentation too aggressive

**Solutions:**
- Increase perceptual loss weight in trainer
- Use pretrained encoder
- Reduce augmentation

### High LPIPS (>0.20)

**Possible Causes:**
- Model generates unrealistic textures
- Lacking perceptual training

**Solutions:**
- Train with multi-layer perceptual loss
- Use pretrained encoder
- Add more training data

---

## 📊 Comparing Multiple Models

### Evaluate Different Checkpoints

```bash
# Evaluate epoch 10
python evaluate.py --checkpoint checkpoints/refinement_epoch_10.pth --test_dir data/test

# Evaluate epoch 20
python evaluate.py --checkpoint checkpoints/refinement_epoch_20.pth --test_dir data/test

# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model.pth --test_dir data/test
```

### Create Comparison Table

```python
import json

checkpoints = ['epoch_10', 'epoch_20', 'best_model']
results = {}

for ckpt in checkpoints:
    with open(f'results/eval_{ckpt}/evaluation_results.json') as f:
        results[ckpt] = json.load(f)

# Print comparison
print(f"{'Model':<15} {'PSNR':<10} {'SSIM':<10}")
print("-" * 35)
for name, data in results.items():
    psnr = data['psnr']['mean']
    ssim = data['ssim']['mean']
    print(f"{name:<15} {psnr:<10.2f} {ssim:<10.4f}")
```

---

## 🎯 Metrics During Training

Metrics are automatically tracked during training:

```bash
# View training history
python -c "
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)
    
    # Find best epoch
    best_epoch = max(range(len(history)), 
                     key=lambda i: history[i].get('val_ssim', 0))
    
    print(f'Best Epoch: {best_epoch + 1}')
    print(f'Best Val SSIM: {history[best_epoch][\"val_ssim\"]:.4f}')
    print(f'Best Val PSNR: {history[best_epoch][\"val_psnr\"]:.2f}')
"
```

### Plot Training Curves

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/training_history.json') as f:
    history = json.load(f)

# Extract metrics
epochs = range(1, len(history) + 1)
train_loss = [h['train_loss'] for h in history]
val_ssim = [h.get('val_ssim', 0) for h in history]

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, train_loss)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

ax2.plot(epochs, val_ssim)
ax2.set_title('Validation SSIM')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('SSIM')
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
```

---

## 🔬 Custom Metrics

### Add Your Own Metric

Edit `src/evaluation/evaluate.py`:

```python
class EvaluationMetrics:
    
    @staticmethod
    def your_metric(pred, target):
        """Your custom metric"""
        # Implementation
        return metric_value
```

Then use in evaluation:

```python
# In evaluate_batch method
results['your_metric'] = self.metrics.your_metric(restored, clean).item()
```

---

## 📚 References

### Metrics Papers

- **PSNR/MSE:** Classic image quality metrics
- **SSIM:** Wang et al. "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
- **LPIPS:** Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (2018)

### Implementation

- SSIM: `pytorch-msssim`
- LPIPS: `lpips` package
- Custom implementations in `src/evaluation/evaluate.py`

---

## ✅ Quick Reference

```bash
# Standard evaluation
python main.py --mode evaluate --test_dir data/test

# Full evaluation with all metrics
python main.py --mode evaluate --test_dir data/test --use_lpips

# Standalone script
python evaluate.py --checkpoint checkpoints/best_model.pth --test_dir data/test

# View training metrics
cat checkpoints/training_history.json

# Check evaluation results
cat results/eval/results_table.md
```

---

**Last Updated:** November 2024
