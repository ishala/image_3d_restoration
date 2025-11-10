# src/recons_pipeline.py
import os
import pycolmap
from .mesh.surface_recons import run_surface_reconstruction
from .mesh.mesh_refinement import run_mesh_refinement
from .texturing.texturing import run_texturing
from .mvs.mvs_dense_recons import run_mvs
from .ai.depth_estimation import run_depth_estimation


def run_pipeline(dataset_paths: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    sparse_dir = dataset_paths["sparse"]
    images_dir = dataset_paths["images"]

    model_dirs = [
        os.path.join(sparse_dir, d)
        for d in os.listdir(sparse_dir)
        if os.path.isdir(os.path.join(sparse_dir, d))
    ]
    if not model_dirs:
        raise RuntimeError("❌ Tidak ada hasil SfM di folder sparse/")

    sfm_model_path = model_dirs[0]
    reconstruction = pycolmap.Reconstruction(sfm_model_path)
    print(f"📂 Loaded reconstruction: {len(reconstruction.points3D)} sparse points.")
    
    depth_output_dir = os.path.join(output_dir, "depth_maps")
    run_depth_estimation(images_dir, depth_output_dir) 

    # === MVS (Dense Reconstruction) ===
    dense_dir = os.path.join(output_dir, "dense")
    dense_ply = run_mvs(sfm_model_path, images_dir, dense_dir, depth_dir=depth_output_dir)

    # === Surface Reconstruction ===
    mesh_dir = os.path.join(output_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, "surface_mesh.ply")
    sparse_dir = os.path.join(sparse_dir, "0")
    run_surface_reconstruction(sparse_dir, mesh_path)

    # === Refinement ===
    refined_mesh_path = os.path.join(mesh_dir, "refined_mesh.ply")
    run_mesh_refinement(mesh_path, refined_mesh_path)

    # === Texturing ===
    textured_mesh_path = os.path.join(output_dir, "textured_mesh.obj")
    run_texturing(refined_mesh_path, images_dir, textured_mesh_path)

    return {
        "dense_point_cloud": dense_ply,
        "mesh": mesh_path,
        "refined_mesh": refined_mesh_path,
        "textured_mesh": textured_mesh_path,
    }
