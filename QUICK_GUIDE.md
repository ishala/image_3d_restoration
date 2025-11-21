# 3D Artifact Restoration - Quick Guide 🚀

Pipeline AI untuk restorasi gambar artefak rusak dan rekonstruksi 3D.

---

## 📦 Setup Environment

### 1. Install Dependencies
```bash
# Aktifkan conda environment
conda activate restore

# Install requirements
pip install -r requirements.txt
```

**Requirements utama:**
- PyTorch 2.0+ (CUDA 11.8)
- Open3D 0.17.0
- pycolmap 0.4.0
- transformers 4.27+

### 2. Verifikasi Installation
```bash
# Test GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test Open3D
python -c "import open3d; print('Open3D:', open3d.__version__)"
```

---

## 🎯 Pipeline Workflow

### **Mode 1: PRETRAIN** (Opsional, tapi Direkomendasikan) ✨

Self-supervised pretraining untuk dataset terbatas.

```bash
python main.py --mode pretrain \
    --pretrain_data data/unlabeled_artifacts \
    --pretrain_task rotation \
    --pretrain_epochs 15 \
    --batch_size 8
```

**Output:** `checkpoints/pretrained_encoder.pth`


---

### **Mode 2: RECONSTRUCT** (Prepare Training Data)

Rekonstruksi 3D dari dataset demo untuk membuat training data.

```bash
# Reconstruct semua artifacts
python main.py --mode reconstruct --dataset data/data_demo

# Reconstruct artifact tertentu
python main.py --mode reconstruct --dataset data/data_demo --artifact greek
```

**Input:** `data/data_demo/[artifact_name]/` (folder berisi multi-view images)

**Output:** `outputs/[artifact_name]/`
- `images/` - Original images
- `sparse/` - SfM reconstruction
- `dense/depth_maps/` - Depth maps (untuk training)
- `dense/mesh/` - 3D mesh

**Waktu:** ~1-5 menit per artifact

---

### **Mode 3: TRAIN** (Train Restoration Model)

Train model untuk restorasi gambar rusak.

```bash
# Train dengan data dari data/train (RECOMMENDED - struktur baru)
python main.py --mode train \
    --use_pretrained_encoder \
    --epochs 50 \
    --batch_size 4

# Train tanpa pretraining (baseline)
python main.py --mode train --epochs 50 --batch_size 4

# Train pada artifact spesifik (old structure dari outputs/)
python main.py --mode train --artifact boy_with_thorn --epochs 30
```

**Input (Struktur Baru - Recommended):** 
```
data/
  train/
    artifact1/
      image1.jpg
      image2.jpg
    artifact2/
      ...
  test/
    artifact1/
      ...
```

**Input (Struktur Lama):** Data dari `outputs/` (hasil reconstruct)

**Output:** `checkpoints/`
- `best_model.pth` - Model terbaik (highest SSIM)
- `last_model.pth` - Model epoch terakhir
- `training_history.json` - Metrics history

**Model Architecture:**
- ResNet50 U-Net backbone
- Input: RGB + Depth (4 channels)
- Loss: 50% L1 + 30% Multi-Layer Perceptual + 20% SSIM
- Automatic damage simulation (cracks, erosion, weathering)

**Waktu:** ~30-60 menit (50 epochs, GPU)

**Notes:**
- Dataset akan otomatis generate damage simulation dari clean images
- Tidak perlu depth maps terpisah (auto-generated)
- Training menggunakan `data/train/`, validation menggunakan `data/test/`

---

### **Mode 4: RETRAIN** (Fine-tune Model)

Lanjutkan training dari checkpoint yang ada.

```bash
# Fine-tune dari best model
python main.py --mode train \
    --use_pretrained_encoder \
    --epochs 20 \
    --batch_size 4

# Model akan auto-load dari checkpoints/last_model.pth
```

**Tips:**
- Gunakan learning rate lebih kecil untuk fine-tuning
- Training akan continue dari checkpoint terakhir
- Metrics akan di-append ke training history

---

## 🔮 Inference (Running Model)

### **A. Single Image Inference** ⚡ (Tercepat)

Restorasi 1 gambar rusak → 2D restored + 3D mesh.

```bash
# Basic usage
python main.py --mode single_inference --input_image damaged_statue.jpg

# With ground truth for quality evaluation
python main.py --mode single_inference \
    --input_image damaged.jpg \
    --ground_truth clean_reference.jpg

# Custom output directory
python main.py --mode single_inference \
    --input_image damaged.jpg \
    --output_dir results/my_artifact \
    --ground_truth clean.jpg
```

**Input:** 1 gambar rusak (JPG/PNG)

**Output:** `inference_results/single/[image_name]/`
- `input.jpg` - Original damaged image
- `restored.jpg` - Restored image (2D)
- `damage_mask.jpg` - Detected damage mask
- `depth_estimated.png` - Estimated depth map
- `mesh_3d.ply` - 3D mesh
- `mesh_3d.obj` - 3D mesh (OBJ format)
- `metrics.json` - Quality metrics (jika ground truth provided)

**Quality Metrics (jika ground truth tersedia):**
```json
{
  "psnr": 28.45,
  "ssim": 0.8912,
  "mae": 0.0345,
  "mse": 0.0089
}
```

**Waktu:** ~5-10 detik (GPU) | ~30 detik (CPU)

---

### **B. Multi-View Inference** 🎯 (Paling Akurat)

Restorasi multiple images → High-quality 3D reconstruction.

```bash
# Basic usage
python main.py --mode multi_inference --input_dir damaged_images/

# With depth estimation
python main.py --mode multi_inference \
    --input_dir damaged_images/ \
    --estimate_depth \
    --output_dir results/artifact_3d
```

**Input:** Folder berisi 10+ images dari berbagai sudut

**Output:** `inference_results/multi/[folder_name]/`
- `restored_images/` - Semua gambar yang sudah direstorasi
- `depth_maps/` - Depth maps (jika --estimate_depth)
- `sparse/` - SfM reconstruction
- `dense/` - Dense point cloud + mesh
- `textured_mesh.ply` - Final 3D model with texture

**Waktu:** ~1-5 menit tergantung jumlah gambar

**Tips:**
- Minimal 10 images, ideal 20-50 images
- 70% overlap antar gambar
- Lighting konsisten
- Berbagai sudut pandang

---

## 👁️ Visualisasi Results

### **1. Visualisasi 2D Results**

View restored images & damage masks:

```bash
# View semua outputs dari inference
python src/visualization/visualize_outputs.py --results_dir inference_results/single/damaged_artifact

# Before/after comparison
python src/visualization/visualize_outputs.py --mode side_by_side
```

---

### **2. Visualisasi 3D Mesh** 🎮

#### **Option A: Quick View (Termudah)**

```bash
# Langsung buka mesh viewer
python -c "import open3d as o3d; mesh = o3d.io.read_triangle_mesh('inference_results/single/damaged_artifact/mesh_3d.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"
```

#### **Option B: Using MeshLab/Blender**

```bash
# Install MeshLab
# Windows: Download dari meshlab.net
# Ubuntu: sudo apt install meshlab

# Open mesh
meshlab inference_results/single/damaged_artifact/mesh_3d.ply
```

#### **Option C: Python Script**

```python
import open3d as o3d

# Load mesh
mesh = o3d.io.read_triangle_mesh("inference_results/single/damaged_artifact/mesh_3d.ply")
mesh.compute_vertex_normals()

# Visualize
o3d.visualization.draw_geometries(
    [mesh],
    window_name="3D Artifact",
    width=1280, height=720,
    mesh_show_back_face=True
)
```

**Controls:**
- **Left Mouse:** Rotate
- **Right Mouse:** Pan
- **Scroll:** Zoom
- **R:** Reset view
- **Q/ESC:** Quit

---

### **3. View Dense Point Cloud**

```bash
python -c "
import open3d as o3d
pcd = o3d.io.read_point_cloud('inference_results/multi/artifact/dense/fused.ply')
o3d.visualization.draw_geometries([pcd])
"
```

---

## 📊 Evaluate Model Performance

### **A. Comprehensive Evaluation** 🎯 (Recommended)

Evaluate model dengan metrik lengkap pada test set.

```bash
# Automatic evaluation (uses data/test/)
python main.py --mode evaluate

# Basic evaluation with custom directory
python main.py --mode evaluate --test_dir data/test

# With LPIPS perceptual metric
python main.py --mode evaluate \
    --test_dir data/test \
    --use_lpips \
    --batch_size 4

# Custom output directory
python main.py --mode evaluate \
    --test_dir data/test \
    --output_dir results/eval_model_v2
```

**Input Structure:**
```
data/test/
  artifact1/
    image1.jpg
    image2.jpg
  artifact2/
    ...
```

**Output:** `results/eval/`
- `evaluation_results.json` - Semua metrik (JSON)
- `results_table.md` - Tabel hasil (Markdown)
- `examples/` - Visual comparisons

**Metrics Evaluated:**
- ✅ **PSNR** - Peak Signal-to-Noise Ratio
- ✅ **SSIM** - Structural Similarity Index
- ✅ **MAE** - Mean Absolute Error
- ✅ **MSE** - Mean Squared Error
- ✅ **LPIPS** - Learned Perceptual Similarity (optional)

**Sample Output:**
```
📊 EVALUATION RESULTS
======================
PSNR:
  Mean:   28.4523
  Std:    2.1234
  Min:    24.5678
  Max:    34.2109
  Median: 28.1234

SSIM:
  Mean:   0.8912
  Std:    0.0456
  ...
```

---

### **B. Quick Evaluation** (Training Metrics)

```bash
# View training history
python -c "
import json
with open('checkpoints/training_history.json') as f:
    history = json.load(f)
    print(f'Best SSIM: {max(h[\"val_ssim\"] for h in history if \"val_ssim\" in h):.4f}')
    print(f'Final Loss: {history[-1][\"train_loss\"]:.4f}')
"
```

---

### **C. Visual Comparison**

```python
# Check inference results
from PIL import Image
import matplotlib.pyplot as plt

# Load images
original = Image.open("damaged.jpg")
restored = Image.open("inference_results/single/damaged/restored.jpg")

# Side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.imshow(original); ax1.set_title("Original (Damaged)")
ax2.imshow(restored); ax2.set_title("Restored")
plt.tight_layout()
plt.show()
```

---

### **D. Evaluation Metrics Explained**

| Metric | Description | Good Value | Range |
|--------|-------------|------------|-------|
| **PSNR** | Signal quality, higher = better | >28 dB | 0-∞ |
| **SSIM** | Structural similarity | >0.85 | 0-1 |
| **MAE** | Average pixel error | <0.05 | 0-1 |
| **MSE** | Squared error | <0.01 | 0-1 |
| **LPIPS** | Perceptual similarity, lower = better | <0.15 | 0-1 |

**Target Metrics:**
- ✅ PSNR: >28 dB
- ✅ SSIM: >0.85
- ✅ LPIPS: <0.15

---

## 📁 Directory Structure

```
Projek 3D Restoration/
├── main.py                          # Main entry point
├── requirements.txt                 # Dependencies
├── data/                            # Dataset (NEW STRUCTURE)
│   ├── train/                       # Training images
│   │   ├── artifact1/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── artifact2/
│   └── test/                        # Test/validation images
│       ├── artifact1/
│       └── artifact2/
├── outputs/                         # Training data (OLD - dari reconstruct)
│   ├── greek/
│   │   ├── images/                  # Original images
│   │   ├── sparse/                  # SfM data
│   │   └── dense/
│   │       ├── depth_maps/          # Depth maps (for training)
│   │       └── mesh/                # 3D meshes
│   └── ...
├── checkpoints/                     # Model checkpoints
│   ├── pretrained_encoder.pth       # Pretrained encoder
│   ├── best_model.pth              # Best trained model
│   ├── last_model.pth              # Latest checkpoint
│   └── training_history.json       # Training metrics
├── inference_results/               # Inference outputs
│   ├── single/                      # Single image results
│   │   └── [image_name]/
│   │       ├── restored.jpg
│   │       ├── mesh_3d.ply
│   │       └── ...
│   └── multi/                       # Multi-view results
│       └── [folder_name]/
│           ├── restored_images/
│           ├── dense/
│           └── textured_mesh.ply
├── results/                         # Evaluation results
│   └── eval/
│       ├── evaluation_results.json  # Metrics (JSON)
│       ├── results_table.md        # Results table
│       └── examples/               # Visual comparisons
└── src/
    ├── ai/                          # AI models & training
    ├── datasets/                    # Dataset loaders
    ├── evaluation/                  # Evaluation metrics & scripts
    │   └── evaluate.py             # Comprehensive evaluation
    ├── sfm/                         # Structure from Motion
    ├── mvs/                         # Multi-View Stereo
    ├── mesh/                        # Mesh generation
    └── visualization/               # Visualization tools
```

---

## 🔧 Common Issues & Solutions

### 1. Out of Memory (OOM)

```bash
# Reduce batch size
python main.py --mode train --batch_size 2  # or 1

# Use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

### 2. COLMAP Not Found

```bash
# Install COLMAP
# Ubuntu: sudo apt install colmap
# Windows: Download from colmap.github.io
```

### 3. Poor 3D Reconstruction

**Solusi:**
- Tambah jumlah images (min 10, ideal 20+)
- Pastikan 70% overlap antar gambar
- Gunakan `--estimate_depth` untuk multi-view
- Lighting konsisten

### 4. Low Quality Restoration

**Solusi:**
- Train lebih lama (50-100 epochs)
- Gunakan pretrained encoder (`--use_pretrained_encoder`)
- Tambah training data (reconstruct lebih banyak artifacts)

---

## 💡 Tips & Best Practices

### Training
✅ Selalu gunakan pretrained encoder untuk dataset terbatas  
✅ Monitor validation SSIM (target: >0.85)  
✅ Save checkpoints tiap 5 epochs  
✅ Use GPU untuk training (50x lebih cepat)

### Inference
✅ Single image: untuk quick preview  
✅ Multi-view: untuk high-quality 3D  
✅ Minimal 10 images untuk multi-view  
✅ Consistent lighting & background

### 3D Reconstruction
✅ Overlap 70% antar gambar  
✅ Berbagai sudut pandang (360°)  
✅ Avoid blurry images  
✅ Consistent scale

---

## 🚀 Quick Commands Cheatsheet

```bash
# === SETUP ===
conda activate restore
pip install -r requirements.txt

# === PRETRAIN (Optional) ===
python main.py --mode pretrain --pretrain_data data/unlabeled --pretrain_epochs 15

# === PREPARE DATA ===
# Option 1: Use existing data/train and data/test (RECOMMENDED)
# Just make sure images are in data/train/artifact_name/ and data/test/artifact_name/

# Option 2: Reconstruct from multi-view (for 3D mesh generation)
python main.py --mode reconstruct --dataset data/data_demo

# === TRAIN ===
python main.py --mode train --use_pretrained_encoder --epochs 50 --batch_size 4

# === EVALUATE ===
python main.py --mode evaluate  # Auto uses data/test/

# === INFERENCE ===
# Single image
python main.py --mode single_inference --input_image damaged.jpg

# Multi-view
python main.py --mode multi_inference --input_dir images/ --estimate_depth

# === VISUALIZE ===
# 3D mesh
python -c "import open3d as o3d; m=o3d.io.read_triangle_mesh('mesh_3d.ply'); m.compute_vertex_normals(); o3d.visualization.draw_geometries([m])"
```

---

## 📚 Further Reading

- `README.md` - Full documentation
- `docs/PHASE2_IMPLEMENTATION.md` - Advanced features
- `docs/INFERENCE_GUIDE.md` - Detailed inference guide

---

**Last Updated:** November 2024  
**Version:** 2.1 (Phase 2: Self-Supervised Learning)
