# src/sfm/sfm_pycolmap.py
import os
import pycolmap
from .utils import *

def run_sfm(dataset_paths: dict):
    """
    Jalankan Structure-from-Motion dengan PyCOLMAP standar (tanpa LoFTR, tanpa augmentasi, tanpa reuse cache).
    
    Args:
        dataset_paths (dict): dict berisi {"images":..., "sparse":..., "dense":...}
    
    Returns:
        list of pycolmap.Reconstruction
    """
    images_dir = dataset_paths["images"]
    sparse_dir = dataset_paths["sparse"]
    db_path = os.path.join(sparse_dir, "database.db")

    # 0. Cek dan buat sparse_dir
    sparse_dir_ok = check_and_create_sparse_folder(sparse_dir)
    # 1. Inisialisasi database
    if sparse_dir_ok:
        check_and_create_db_folder(db_path)

    db = pycolmap.Database(db_path)

    # 2. Deteksi GPU
    use_gpu = False
    gpu_info = None
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            gpu_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} device)"
    except Exception:
        pass

    if use_gpu:
        print(f"🔧 GPU terdeteksi. Info: {gpu_info} — akan menggunakan GPU untuk ekstraksi fitur.")
    else:
        print("ℹ️ GPU tidak terdeteksi — akan menggunakan CPU untuk ekstraksi fitur.")

    # 3. Ekstraksi fitur SIFT
    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.use_gpu = use_gpu
    sift_opts.num_threads = 4
    sift_opts.max_image_size = 1600

    device = pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu

    try:
        pycolmap.extract_features(
            database_path=db_path,
            image_path=images_dir,
            sift_options=sift_opts,
            device=device
        )
        print("✅ Ekstraksi fitur SIFT selesai.")
    except Exception as e:
        print(f"❌ Error saat ekstraksi fitur SIFT: {e}")
        return None

    # 4. Matching fitur (exhaustive)
    try:
        pycolmap.match_exhaustive(db_path)
        print("✅ Matching fitur selesai.")
    except Exception as e:
        print(f"❌ Error saat matching fitur: {e}")
        return None

    # 5. Jalankan mapping (rekonstruksi 3D)
    try:
        maps = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=images_dir,
            output_path=sparse_dir
        )
        print(f"✅ SfM selesai, total {len(maps)} model dibuat di {sparse_dir}")
        return maps
    except Exception as e:
        print(f"❌ Error dalam incremental mapping: {e}")
        return None
