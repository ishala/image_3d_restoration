import os
from PIL import Image
import numpy as np
import torch
import open3d as o3d

# pip install transformers timm torch torchvision  (sesuaikan versi torch dengan CUDA/CPU)
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# --- Device helper dan model cache ---
_MODEL_CACHE = {}

def get_dpt_model(model_name="Intel/dpt-hybrid-midas", device=None):
    """
    Load model & feature extractor (cached).
    Default: "Intel/dpt-small" (lebih ringan). 
    Alternatives: "Intel/dpt-large" (lebih akurat, lebih besar).
    """
    global _MODEL_CACHE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = f"{model_name}@{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    fe = DPTFeatureExtractor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name).to(device)
    model.eval()
    _MODEL_CACHE[key] = (fe, model, device)
    return _MODEL_CACHE[key]

def estimate_depth(image_path, model_name="Intel/dpt-hybrid-midas", device=None):
    """
    Return depth (numpy float32) normalized (0..1).
    """
    fe, model, device = get_dpt_model(model_name, device)
    image = Image.open(image_path).convert("RGB")
    inputs = fe(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth[0].cpu().numpy()
    # normalize to 0..1 for convenience (keep relative)
    depth = depth.astype("float32")
    if np.max(depth) > 0:
        depth = depth / np.max(depth)
    return depth

# ---------------------------
# Utilities: depth -> point cloud
# ---------------------------
def depth_to_pointcloud(depth_map, color_image=None, intrinsics=None, depth_scale=1.0, stride=1):
    """
    Convert depth map (H,W) normalized or relative to Open3D PointCloud.
    - depth_map: numpy HxW (float), expected proportional distances (relative). 
    - color_image: PIL or np array HxW x3 (optional) for colors.
    - intrinsics: dict or numpy 3x3 camera intrinsics matrix. If None, use simple pinhole with fx=fy= W/ (2*tan(fov/2))
    - depth_scale: multiplier to scale relative depth to workable metric range.
    - stride: downsample step to reduce points.
    Returns: open3d.geometry.PointCloud
    """
    import math
    import cv2
    import numpy as np
    import open3d as o3d
    from PIL import Image

    h, w = depth_map.shape

    # ✅ --- FIX: pastikan ukuran color_image sama dengan depth_map ---
    if color_image is not None:
        if isinstance(color_image, Image.Image):
            color_image = np.array(color_image)
        if color_image.shape[:2] != (h, w):
            color_image = cv2.resize(color_image, (w, h), interpolation=cv2.INTER_LINEAR)
        color_image = color_image.astype(np.float32) / 255.0
    # ✅ --- END FIX ---

    # Tentukan intrinsics
    if intrinsics is None:
        fx = fy = max(w, h)
        cx = w / 2.0
        cy = h / 2.0
    else:
        if isinstance(intrinsics, np.ndarray) and intrinsics.shape == (3, 3):
            fx = intrinsics[0, 0]; fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]; cy = intrinsics[1, 2]
        else:
            fx = intrinsics.get("fx", max(w, h))
            fy = intrinsics.get("fy", max(w, h))
            cx = intrinsics.get("cx", w / 2.)
            cy = intrinsics.get("cy", h / 2.)

    zs = depth_map * depth_scale
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    zs_flat = zs[ys, xs]

    mask = np.isfinite(zs_flat) & (zs_flat > 1e-6)
    xs = xs[mask]
    ys = ys[mask]
    zs_flat = zs_flat[mask]

    x_cam = (xs - cx) * zs_flat / fx
    y_cam = (ys - cy) * zs_flat / fy

    pts = np.vstack((x_cam, -y_cam, zs_flat)).T  # Z-forward, flip y

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))

    if color_image is not None:
        cols = color_image[ys, xs, :]
        pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float32))

    return pcd

# ---------------------------
# Helper: save depth & pcd
# ---------------------------
def save_depth(depth, out_path_png):
    from PIL import Image
    d = (255 * (depth - depth.min()) / (depth.max()-depth.min()+1e-9)).astype("uint8")
    Image.fromarray(d).save(out_path_png)

def save_pcd(pcd, out_path_ply):
    o3d.io.write_point_cloud(out_path_ply, pcd)
    

def run_depth_estimation(images_dir: str, output_dir: str, model_name: str = "Intel/dpt-hybrid-midas", device: str = 'cpu'):
    """
    Jalankan depth estimation untuk semua gambar dalam folder.
    Simpan hasil depth map (.png) dan point cloud (.ply) ke output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))]

    for fname in image_files:
        img_path = os.path.join(images_dir, fname)
        base = os.path.splitext(fname)[0]

        print(f"[AI Depth] Estimating depth for {fname} ...")
        depth = estimate_depth(img_path, model_name=model_name)
        color_img = Image.open(img_path).convert("RGB")

        # save depth map
        depth_png_path = os.path.join(output_dir, f"{base}_depth.png")
        save_depth(depth, depth_png_path)

        # convert to point cloud
        pcd = depth_to_pointcloud(depth, color_image=color_img, depth_scale=1.0, stride=4)
        pcd_path = os.path.join(output_dir, f"{base}_depth.ply")
        save_pcd(pcd, pcd_path)

        print(f"  ✔ saved depth map: {depth_png_path}")
        print(f"  ✔ saved point cloud: {pcd_path}")

    print(f"[AI Depth] Done. Total processed: {len(image_files)} images.")
    return output_dir