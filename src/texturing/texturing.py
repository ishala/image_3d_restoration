# src/reconstructions/texturing.py
import os
import open3d as o3d
import numpy as np
from PIL import Image

def run_texturing(mesh_path: str, images_dir: str, output_path: str):
    """
    Memberikan warna (texturing) pada mesh hasil rekonstruksi 3D.
    
    Args:
        mesh_path (str): Path ke mesh hasil refinement (PLY/OBJ)
        images_dir (str): Folder berisi gambar input asli
        output_path (str): File output textured mesh (format OBJ recommended)
    """
    print("🎨 Memulai proses texturing...")

    # Load mesh hasil refinement
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError("❌ Mesh tidak bisa dibuka atau kosong!")

    # Estimasi normal (kalau belum ada)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Ambil semua gambar input
    image_files = [os.path.join(images_dir, f) 
                   for f in os.listdir(images_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(image_files) == 0:
        raise RuntimeError("❌ Tidak ditemukan gambar untuk texturing.")

    # Load satu gambar untuk estimasi warna dasar (opsional)
    ref_img = np.asarray(Image.open(image_files[0]).convert("RGB"), dtype=np.float32) / 255.0

    # Warna mesh berdasarkan intensitas rata-rata
    color_mean = ref_img.reshape(-1, 3).mean(axis=0)
    mesh.paint_uniform_color(color_mean)

    # Simpan hasil mesh berwarna
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"✅ Texturing selesai -> {output_path}")

    return output_path
