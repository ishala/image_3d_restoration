# src/mesh/surface_recons.py
import os
import numpy as np
import open3d as o3d
import pycolmap
from typing import Union

def _extract_points_and_colors_from_reconstruction(reconstruction):
    pts = []
    cols = []
    for p in reconstruction.points3D.values():
        # Ambil koordinat
        if hasattr(p, "xyz"):
            xyz = np.asarray(p.xyz, dtype=np.float64)
        else:
            # fallback: kalau struktur lain, coba index 0
            try:
                xyz = np.asarray(p[0], dtype=np.float64)
            except Exception:
                continue
        pts.append(xyz)

        # Ambil warna (jika ada)
        if hasattr(p, "rgb"):
            rgb = np.asarray(p.rgb, dtype=np.float64) / 255.0
            cols.append(rgb)
        elif hasattr(p, "color"):
            rgb = np.asarray(p.color, dtype=np.float64) / 255.0
            cols.append(rgb)
        else:
            cols.append([0.5, 0.5, 0.5])  # default gray

    pts = np.asarray(pts)
    cols = np.asarray(cols) if len(cols) == len(pts) else None
    return pts, cols


def run_surface_reconstruction(sfm_input: Union[str, pycolmap.Reconstruction],
                               output_mesh_path: str,
                               method: str = "poisson",
                               poisson_depth: int = 9):
    """
    Buat surface mesh dari hasil SfM.
    sfm_input: path ke folder sparse (string) ATAU objek pycolmap.Reconstruction
    output_mesh_path: path file .ply yang akan disimpan
    method: "poisson" atau "bpa"
    """
    # pastikan folder output ada
    out_dir = os.path.dirname(output_mesh_path)
    os.makedirs(out_dir, exist_ok=True)

    # load reconstruction jika input string
    if isinstance(sfm_input, str):
        reconstruction = pycolmap.Reconstruction(sfm_input)
    else:
        reconstruction = sfm_input

    pts, cols = _extract_points_and_colors_from_reconstruction(reconstruction)

    if pts.size == 0:
        raise RuntimeError("❌ Tidak ada point3D yang bisa diekstrak dari reconstruction.")

    # konversi ke Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd.colors = o3d.utility.Vector3dVector(cols)

    print(f"[surface_recons] Loaded {len(pts)} points from reconstruction")

    # normal estimation (penting sebelum Poisson)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    if method == "poisson":
        print(f"[surface_recons] Running Poisson (depth={poisson_depth}) ...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
        # crop with pcd bbox to remove far-away blobs
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
    elif method == "bpa":
        print("[surface_recons] Running Ball Pivoting Algorithm ...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances))
        radii = o3d.utility.DoubleVector([avg_dist * 1.5, avg_dist * 3.0])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    else:
        raise ValueError("method must be 'poisson' or 'bpa'")

    # basic cleaning
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    # write mesh
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print(f"[surface_recons] Mesh saved to: {output_mesh_path}")

    return output_mesh_path