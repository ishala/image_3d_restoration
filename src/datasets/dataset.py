import os
import shutil
from .augmentations import augment_dataset


def initial_done(output_root):
    done_output_path = os.path.join(output_root, 'done.txt')
    if os.path.exists(done_output_path):
        with open(done_output_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        text = ''
    done_outputs = [s.strip() for s in text.split(',') if s != '']
    return done_outputs

def prepare_dataset_structure(dataset_dir, output_dir, is_augment, excluded_aug=None):
    """
    Siapkan struktur dataset sesuai format Pycolmap:
      - output_dir/images
      - output_dir/sparse
      - output_dir/dense
    
    Args:
        dataset_dir: Path ke direktori dataset
        output_dir: Path ke direktori output
        is_augment: Boolean flag untuk augmentasi
        excluded_datasets: List nama dataset yang tidak perlu diaugmentasi
    """
    if excluded_aug is None:
        excluded_excluded_aug = ['boy_with_thorn']
        
    # Tentukan apakah dataset ini perlu diaugmentasi
    dataset_name = os.path.basename(dataset_dir)
    should_augment = is_augment and dataset_name not in excluded_aug
    
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")

    for d in [images_dir, sparse_dir, dense_dir]:
        os.makedirs(d, exist_ok=True)

    # Hapus file augmentasi yang mungkin ada dari run sebelumnya
    if is_augment:
        for file_name in os.listdir(images_dir):
            if file_name.startswith("aug_"):
                file_path = os.path.join(images_dir, file_name)
                try:
                    os.remove(file_path)
                    print(f"Berhasil hapus {file_path}")
                except OSError as e:
                    print(f"⚠️ Gagal menghapus file augmentasi lama {file_path}: {e}")

    # copy gambar
    orig_image_count = 0
    for file_name in os.listdir(dataset_dir):
        src_path = os.path.join(dataset_dir, file_name)
        if os.path.isfile(src_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            dst_path = os.path.join(images_dir, file_name)
            shutil.copy2(src_path, dst_path)
            orig_image_count += 1

    # Lakukan augmentasi jika:
    # 1. is_augment=True dan dataset tidak dalam daftar exclude
    # 2. Belum ada file augmentasi
    # if should_augment:
    #     aug_files = [f for f in os.listdir(images_dir) if f.startswith("aug_")]
    #     if not aug_files:
    #         print(f"🔄 Mengaugmentasi {orig_image_count} gambar di {images_dir}...")
    #         augment_dataset(images_dir=images_dir, num_aug=2)
    #     else:
    #         print(f"ℹ️ File augmentasi sudah ada di {images_dir}, skip augmentasi")
    # else:
    #     if dataset_name in excluded_aug:
    #         print(f"ℹ️ Dataset {dataset_name} dalam daftar exclude, hanya melakukan copy tanpa augmentasi")
    #     else:
    #         print(f"ℹ️ Augmentasi dinonaktifkan untuk {dataset_name}")

    return {
        "images": images_dir,
        "sparse": sparse_dir,
        "dense": dense_dir,
    }


def get_subdatasets(root_dir, output_root="outputs", 
                    is_augment: bool = True, 
                    done_outputs: list = None,
                    excluded_aug: list = None):
    """
    Ambil semua subfolder dataset dalam root_dir lalu auto-prepare
    strukturnya ke output_root.
    
    Args:
        root_dir: Path ke root directory dataset
        output_root: Path ke direktori output (default: "outputs")
        is_augment: Flag untuk mengaktifkan augmentasi (default: True)
        done_outputs: List dataset yang sudah selesai (default: None)
        excluded_datasets: List dataset yang tidak perlu diaugmentasi (default: None)
    
    Returns:
        dict: mapping {nama_dataset: paths_dict}
              contoh: {"arc_of_severus": {"images":..., "sparse":..., "dense":...}, ...}
    """
    if done_outputs is None:
        done_outputs = []
    if excluded_aug is None:
        excluded_aug = ['boy_with_thorn']

    subfolders = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    
    # Filter dataset yang sudah selesai
    subfolders_fix = [sub for sub in subfolders if sub not in done_outputs]
    
    results = {}
    for sub in subfolders_fix:
        dataset_path = os.path.join(root_dir, sub)
        output_dir = os.path.join(output_root, sub)
        # Jika output_dir sudah ada, hapus dulu agar prepare_dataset_structure
        # selalu membuat struktur yang bersih. Gunakan shutil.rmtree agar
        # direktori yang berisi file juga bisa dihapus.
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"⚠️ Gagal menghapus {output_dir}: {e}")
        paths = prepare_dataset_structure(dataset_path, output_dir, is_augment, excluded_aug)
        results[sub] = paths
        print(f"✅ Prepared dataset {sub} at {output_dir}")
    
    return results