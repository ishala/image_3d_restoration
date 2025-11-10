import os

def check_and_create_sparse_folder(sparse_dir):
    sparse_dir_ok = os.path.exists(sparse_dir)
    if not sparse_dir_ok:
        print(f"Folder {sparse_dir} not found, then create")
        os.makedirs(sparse_dir, exist_ok=True)
        print(f"Now {sparse_dir} are available")
    else:
        print(f"Making {sparse_dir} failed")
    return sparse_dir_ok

def check_and_create_db_folder(db_path):
    # Bersihkan database lama jika ada
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"🧹 Menghapus database lama: {db_path}")
        except Exception as e:
            print(f"⚠️ Gagal menghapus database lama: {e}")