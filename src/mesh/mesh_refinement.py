import open3d as o3d
import os

def run_mesh_refinement(mesh_path, refined_path, 
                        target_reduction=0.5, smoothing_iter=10, 
                        remove_noise=True, show_process=True):
    """
    Lakukan pembersihan, simplifikasi, dan smoothing pada mesh hasil surface reconstruction.

    Args:
        mesh_path (str): path ke file mesh input (.ply / .obj).
        refined_path (str): path untuk menyimpan hasil refined mesh.
        target_reduction (float): rasio reduksi vertex (0.5 = kurangi 50%).
        smoothing_iter (int): jumlah iterasi smoothing Laplacian.
        remove_noise (bool): apakah hapus noise kecil di permukaan.
        show_process (bool): tampilkan visualisasi sebelum/sesudah.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"❌ File mesh tidak ditemukan: {mesh_path}")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError("❌ Mesh tidak valid atau kosong.")
    print(f"📦 Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # Step 1: Optional — hapus noise dari permukaan
    if remove_noise:
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_non_manifold_edges()
        print("🧹 Noise & degenerate geometry removed.")

    # Step 2: Simplifikasi mesh (quadric decimation)
    if target_reduction < 1.0:
        target_count = int(len(mesh.vertices) * target_reduction)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_count)
        print(f"🔻 Simplified mesh → {len(mesh.vertices)} vertices.")

    # Step 3: Smoothing permukaan
    if smoothing_iter > 0:
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=smoothing_iter)
        print(f"✨ Smoothed mesh ({smoothing_iter} iterations).")

    # Step 4: Recompute normal untuk rendering dan texturing berikutnya
    mesh.compute_vertex_normals()

    # Step 5: Save hasil refinement
    o3d.io.write_triangle_mesh(refined_path, mesh)
    print(f"💾 Refined mesh disimpan di: {refined_path}")

    return refined_path
