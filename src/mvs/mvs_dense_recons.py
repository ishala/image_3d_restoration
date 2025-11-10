import os
import subprocess
import shutil

def run_mvs(sfm_model_path: str, images_dir: str, output_dir: str, depth_dir: str = None):
    """
    Jalankan dense reconstruction (MVS) menggunakan perintah COLMAP CLI.
    Jika depth_dir diberikan, maka hasil depth estimation AI akan diinject
    ke dalam workspace COLMAP sebagai depth prior.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("🧠 Running dense reconstruction (MVS) via COLMAP...")

    # === Step 1: Undistort images ===
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", sfm_model_path,
        "--output_path", output_dir,
        "--output_type", "COLMAP",
    ], check=True)

    # === [Optional] Inject depth maps hasil AI ===
    if depth_dir is not None and os.path.exists(depth_dir):
        target_depth_dir = os.path.join(output_dir, "stereo", "depth_maps")
        os.makedirs(target_depth_dir, exist_ok=True)

        injected_files = 0
        for f in os.listdir(depth_dir):
            if f.lower().endswith((".png", ".exr", ".pfm")):
                shutil.copy(os.path.join(depth_dir, f), os.path.join(target_depth_dir, f))
                injected_files += 1

        print(f"📤 Injected {injected_files} AI depth maps into: {target_depth_dir}")
    else:
        print("⚠️ No AI depth maps provided — running pure COLMAP MVS.")

    # === Step 2: Stereo Matching ===
    patch_match_args = [
        "colmap", "patch_match_stereo",
        "--workspace_path", output_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true"
    ]

    # Jika depth map hasil AI disediakan, tambahkan flag agar COLMAP lebih ringan
    # if depth_dir is not None:
    #     patch_match_args += [
    #         "--PatchMatchStereo.cache_size", "32",
    #         "--PatchMatchStereo.skip_geometric_consistency", "true"
    #     ]

    subprocess.run(patch_match_args, check=True)

    # === Step 3: Stereo Fusion ===
    dense_ply = os.path.join(output_dir, "dense.ply")
    subprocess.run([
        "colmap", "stereo_fusion",
        "--workspace_path", output_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", dense_ply
    ], check=True)

    print(f"✅ Dense point cloud saved at: {dense_ply}")
    return dense_ply
