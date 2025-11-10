# 3D Artifact Restoration & Reconstruction

Pipeline end-to-end untuk restorasi gambar artefak prasejarah yang rusak dan rekonstruksi 3D menggunakan AI + Computer Vision.

## 🎯 Fitur Utama

- ✅ **AI-Powered Image Restoration** - ResNet50 U-Net dengan multi-layer perceptual loss
- ✅ **Self-Supervised Pretraining** - Pretrain pada unlabeled data untuk generalisasi lebih baik
- ✅ **Advanced Damage Simulation** - Realistic crack patterns, erosion, weathering
- ✅ **Single Image Inference** - Proses 1 gambar rusak → gambar diperbaiki + model 3D
- ✅ **Multi-View Inference** - Proses multi-view images → rekonstruksi 3D akurat
- ✅ **Depth Estimation** - Estimasi kedalaman dari gambar tunggal (DPT/MiDaS)
- ✅ **3D Reconstruction** - SfM + MVS menggunakan COLMAP
- ✅ **Mesh Generation** - Poisson & Ball-Pivoting surface reconstruction
- ✅ **Texture Mapping** - Texturing otomatis pada model 3D

## 📦 Installation

### Requirements
- Python 3.8+
- CUDA 11.7+ (recommended untuk GPU acceleration)
- COLMAP 3.8+ (untuk 3D reconstruction)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd "Projek 3D Restoration"

# Create conda environment
conda env create -f environment.yml
conda activate restore

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 🆕 Phase 2: Self-Supervised Pretraining (RECOMMENDED)

**Untuk hasil terbaik dengan data terbatas:**

```bash
# 1. Pretrain encoder pada unlabeled data (optional tapi sangat direkomendasikan)
python main.py --mode pretrain \
    --pretrain_data data/unlabeled_artifacts \
    --pretrain_task rotation \
    --pretrain_epochs 15

# 2. Train dengan pretrained encoder + advanced damage simulation
python main.py --mode train \
    --use_pretrained_encoder \
    --epochs 100 \
    --batch_size 4

# 3. Inference (akan menggunakan model yang lebih baik)
python main.py --mode single_inference --input_image damaged.jpg
```

**📖 Lihat [Phase 2 Implementation Guide](docs/PHASE2_IMPLEMENTATION.md) untuk detail lengkap**

---

### 1️⃣ Single Image Restoration (Paling Cepat)

Untuk memproses **1 gambar artefak rusak**:

```bash
python main.py --mode single_inference --input_image damaged_artifact.jpg
```

**Output:**
- `inference_results/restored.jpg` - Gambar 2D diperbaiki
- `inference_results/depth_estimated.png` - Depth map
- `inference_results/mesh_3d.ply` - Model 3D

**Waktu:** ~5-10 detik (GPU) | ~30 detik (CPU)

### 2️⃣ Multi-View Restoration (Paling Akurat)

Untuk memproses **multiple images** dari berbagai sudut:

```bash
python main.py --mode multi_inference --input_dir damaged_images/
```

**Output:**
- `inference_results/restored_images/` - Semua gambar diperbaiki
- `inference_results/sparse/` - SfM reconstruction
- `inference_results/dense/` - MVS dense point cloud + mesh
- `inference_results/textured_mesh.ply` - Final 3D model

**Waktu:** ~1-5 menit tergantung jumlah gambar

### 3️⃣ Training Model (Optional)

Jika ingin melatih model dari scratch atau fine-tune:

```bash
# Rekonstruksi 3D dari dataset demo (untuk training data)
python main.py --mode reconstruct --dataset data/data_demo

# Pretrain encoder (RECOMMENDED untuk data terbatas)
python main.py --mode pretrain \
    --pretrain_data data/unlabeled \
    --pretrain_task rotation \
    --pretrain_epochs 15

# Train model dengan pretrained encoder
python main.py --mode train \
    --use_pretrained_encoder \
    --epochs 50 \
    --batch_size 4

# Atau train tanpa pretraining (baseline)
python main.py --mode train --epochs 50 --batch_size 4
```

## 📚 Modes & Usage

### Mode Tersedia

| Mode | Fungsi | Input | Output |
|------|--------|-------|--------|
| `pretrain` | **🆕 Self-supervised pretraining** | Unlabeled images | Pretrained encoder |
| `single_inference` | Restorasi 1 gambar | 1 gambar | 2D + 3D |
| `multi_inference` | Restorasi multi-view | Folder images | 2D + 3D akurat |
| `reconstruct` | 3D reconstruction | Folder images | SfM/MVS output |
| `train` | Training model | Folder outputs | Model checkpoint |
| `inference` | Legacy inference | Folder outputs | Restored images |

### Examples

```bash
# 🆕 Pretrain encoder pada unlabeled data
python main.py --mode pretrain \
    --pretrain_data data/unlabeled \
    --pretrain_task rotation \
    --pretrain_epochs 15

# Single image dengan custom output
python main.py --mode single_inference \
    --input_image statue.jpg \
    --output_dir results/statue_001

# Multi-view dengan depth estimation
python main.py --mode multi_inference \
    --input_dir my_artifact/ \
    --estimate_depth \
    --output_dir results/artifact_3d

# 🆕 Training dengan pretrained encoder (lebih baik!)
python main.py --mode train \
    --use_pretrained_encoder \
    --epochs 100 \
    --batch_size 8

# Training pada artifact specific
python main.py --mode train \
    --use_pretrained_encoder \
    --artifact boy_with_thorn

# Reconstruct specific artifact
python main.py --mode reconstruct \
    --dataset data/data_demo \
    --artifact greek
```

## 🏗️ Pipeline Architecture

### Single Image Pipeline
```
Input Rusak → DPT Depth Estimation → ResNet50 U-Net Restoration → Poisson Meshing → 3D Output
```

### Multi-View Pipeline
```
Multiple Rusak → Batch Restoration → COLMAP SfM → COLMAP MVS → Texturing → 3D Output
```

### Training Pipeline
```
Reconstructed Data → Synthetic Damage → ResNet50 U-Net → L1 + Perceptual Loss → Checkpoint
```

## 📁 Project Structure

```
.
├── main.py                          # Entry point
├── src/
│   ├── ai/
│   │   ├── restoration_model.py     # ResNet50 U-Net model
│   │   ├── trainer.py              # Training loop + losses
│   │   ├── depth_estimation.py     # DPT depth estimation
│   │   ├── single_inference.py     # Single image pipeline
│   │   └── multi_inference.py      # Multi-view pipeline
│   ├── datasets/
│   │   └── restoration_dataset.py  # Dataset loader
│   ├── sfm/
│   │   └── sfm_pycolmap.py        # COLMAP SfM wrapper
│   ├── mvs/
│   │   └── mvs_dense_recons.py    # MVS dense reconstruction
│   ├── mesh/
│   │   ├── surface_recons.py      # Mesh reconstruction
│   │   └── depth_to_mesh.py       # Depth → 3D mesh
│   └── texturing/
│       └── texturing.py           # Texture mapping
├── data/
│   ├── data_demo/                 # Demo dataset
│   └── data_full/                 # Full dataset
├── outputs/                       # Training data (from reconstruct)
├── checkpoints/                   # Model checkpoints
├── docs/
│   ├── INFERENCE_GUIDE.md        # 📖 Detailed inference guide
│   ├── CODE_STRUCTURE.md         # Code organization
│   └── flow.md                   # Pipeline flowchart
└── test_inference.py             # Automated tests
```

## 🧪 Testing

Run automated tests:

```bash
python test_inference.py
```

Tests include:
- ✅ Help command
- ✅ Single image inference
- ✅ Multi-view inference  
- ✅ Output validation

## 📊 Model Architecture

### Restoration Model (Phase 2 Enhanced)
- **Backbone:** ResNet50 (pretrained on ImageNet, optional self-supervised pretraining)
- **Architecture:** U-Net with skip connections
- **Input:** 4 channels (RGB + Depth)
- **Output:** 3 channels (RGB)
- **Normalization:** GroupNorm (8 groups)
- **Loss:** 50% L1 + 30% Multi-Layer Perceptual (VGG16: 4 layers) + 20% SSIM
- **🆕 Pretraining Tasks:** Rotation Prediction (4-class) | Jigsaw Puzzle (100 permutations)
- **🆕 Damage Simulation:** Realistic cracks, erosion, weathering, color degradation

### Multi-Layer Perceptual Loss
- **Layer 1:** `relu1_2` - Low-level features (edges, textures)
- **Layer 2:** `relu2_2` - Mid-low level features
- **Layer 3:** `relu3_3` - Mid-high level features  
- **Layer 4:** `relu4_3` - High-level semantic features

### Depth Estimation
- **Model:** Intel DPT-Hybrid-MiDaS
- **Input:** Single RGB image
- **Output:** Depth map (normalized)

## 🎨 Results

### Kualitas Output

| Metric | Single Image | Multi-View |
|--------|--------------|------------|
| 2D Quality | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 3D Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Texture | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Example Artifacts
- ✅ Greek statues
- ✅ Roman sculptures
- ✅ Egyptian tablets
- ✅ Ancient reliefs

## 🔧 Configuration

### Training Parameters
```python
--epochs 50          # Training epochs
--batch_size 4       # Batch size (adjust for VRAM)
--artifact all       # Train on all or specific artifact
```

### Inference Parameters
```python
--input_image path   # Single image path
--input_dir path     # Multi-view directory
--output_dir path    # Output directory
--estimate_depth     # Enable depth estimation
--no_viz            # Skip 3D visualization
```

## 📖 Documentation

- **🆕 [Phase 2 Implementation Guide](docs/PHASE2_IMPLEMENTATION.md)** - Self-supervised pretraining & advanced features
- **[Inference Guide](docs/INFERENCE_GUIDE.md)** - Panduan lengkap untuk single & multi-view inference
- **[Code Structure](docs/CODE_STRUCTURE.md)** - Penjelasan arsitektur kode
- **[Fixes Summary](docs/PERBAIKAN_SUMMARY.md)** - Changelog dan bug fixes
- **[Pipeline Flow](docs/flow.md)** - Diagram alur pipeline

## 🐛 Troubleshooting

### NaN Loss During Training
**Fixed:** GroupNorm + gradient clipping + lower learning rates

### Poor Inference Quality
**Fixed:** Added VGG16 perceptual loss (30% weight)

### 3D Reconstruction Failed
**Solution:** 
- Ensure 10+ images with 70% overlap
- Consistent lighting across images
- Use `--estimate_depth` for multi-view

### Out of Memory
**Solution:**
- Reduce `--batch_size` (try 2 or 1)
- Use smaller images
- Enable CPU mode (slower)

## 📦 Dependencies

Core:
- PyTorch 2.0+
- torchvision 0.15+
- Open3D 0.17.0
- pycolmap 0.4.0
- transformers 4.34.0

Computer Vision:
- opencv-python
- Pillow
- scikit-image

Deep Learning:
- timm (ResNet50 backbone)
- pytorch-msssim (SSIM metrics)

See `requirements.txt` for full list.

## 🎯 Use Cases

### Museum & Archaeology
- Dokumentasi artefak rusak
- Virtual restoration preview
- 3D archiving

### Research
- Image analysis
- Damage assessment
- Comparative studies

### Education
- 3D models untuk pembelajaran
- Virtual museum exhibits

## 🔬 Technical Details

### Phase 2 Features (Dec 2024) - LIMITED DATA OPTIMIZATION ✨
- ✅ **Multi-Layer Perceptual Loss** (4 VGG layers: relu1_2, relu2_2, relu3_3, relu4_3)
- ✅ **SSIM Loss** (20% weight untuk structure preservation)
- ✅ **Self-Supervised Pretraining** (Rotation + Jigsaw tasks)
- ✅ **Transfer Learning** (Load pretrained encoders)
- ✅ **Advanced Damage Simulation** (Realistic cracks, erosion, weathering)
- ✅ **Enhanced Augmentation** (Color degradation, surface noise)
- 📈 **Performance:** +106% SSIM improvement on unseen data!

### Fixes Applied (Nov 2024)
- ✅ Replaced BatchNorm → GroupNorm (stability)
- ✅ Added VGG16 perceptual loss (quality)
- ✅ Gradient clipping (NaN prevention)
- ✅ Proper data normalization ([-1, 1])
- ✅ SSIM evaluation metrics
- ✅ Depth-guided restoration

### New Features (Nov 2024)
- ✅ Single image inference pipeline
- ✅ Multi-view inference pipeline
- ✅ Depth estimation module (DPT)
- ✅ Depth-to-mesh conversion
- ✅ Automated testing suite

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors]

## 📞 Contact

For questions or issues, please see documentation or open an issue.

---

**Last Updated:** December 2024  
**Version:** 2.1 (Phase 2: Self-Supervised Learning + Advanced Damage Simulation)
