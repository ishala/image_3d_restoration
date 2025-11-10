"""
Depth to 3D Mesh Conversion
Convert depth maps and RGB images to textured 3D meshes
"""
import numpy as np
import open3d as o3d
from PIL import Image
import cv2


def depth_to_point_cloud(depth_map, rgb_image, intrinsics=None, depth_scale=1.0, max_depth=10.0):
    """
    Convert depth map + RGB to colored point cloud
    
    Args:
        depth_map: numpy array [H, W] normalized depth (0-1)
        rgb_image: PIL Image or numpy array [H, W, 3]
        intrinsics: dict with fx, fy, cx, cy or None (auto-estimate)
        depth_scale: scale factor for depth values
        max_depth: maximum depth threshold
    
    Returns:
        o3d.geometry.PointCloud
    """
    # Convert inputs
    if isinstance(rgb_image, Image.Image):
        rgb_image = np.array(rgb_image)
    
    h, w = depth_map.shape
    
    # Resize RGB to match depth if needed
    if rgb_image.shape[:2] != (h, w):
        rgb_image = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Setup camera intrinsics
    if intrinsics is None:
        # Auto-estimate focal length
        fov = 60  # degrees
        fx = fy = w / (2 * np.tan(np.radians(fov) / 2))
        cx = w / 2.0
        cy = h / 2.0
    else:
        fx = intrinsics.get('fx', w)
        fy = intrinsics.get('fy', h)
        cx = intrinsics.get('cx', w / 2.0)
        cy = intrinsics.get('cy', h / 2.0)
    
    # Create meshgrid
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Scale depth
    z = depth_map * depth_scale
    
    # Unproject to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb_image.reshape(-1, 3) / 255.0
    
    # Filter invalid points
    valid_mask = (z.flatten() > 0) & (z.flatten() < max_depth) & np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    
    return pcd


def point_cloud_to_mesh(pcd, method='poisson', depth=9):
    """
    Convert point cloud to mesh
    
    Args:
        pcd: o3d.geometry.PointCloud
        method: 'poisson' or 'ball_pivoting'
        depth: octree depth for Poisson (higher = more detail)
    
    Returns:
        o3d.geometry.TriangleMesh
    """
    print(f"📐 Converting point cloud to mesh using {method}...")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)
    
    if method == 'poisson':
        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low density vertices (outliers)
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.01)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
    elif method == 'ball_pivoting':
        # Ball pivoting algorithm
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * r for r in [0.5, 1.0, 1.5, 2.0]]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Post-processing
    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    print(f"✅ Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh


def create_textured_mesh(depth_map, rgb_image, intrinsics=None, method='poisson', depth_scale=1.0):
    """
    End-to-end: depth + RGB → textured mesh
    
    Args:
        depth_map: numpy array [H, W]
        rgb_image: PIL Image or numpy array [H, W, 3]
        intrinsics: camera intrinsics dict
        method: reconstruction method
        depth_scale: depth scaling factor
    
    Returns:
        o3d.geometry.TriangleMesh with vertex colors
    """
    # Step 1: Create point cloud
    pcd = depth_to_point_cloud(depth_map, rgb_image, intrinsics, depth_scale)
    
    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Step 2: Convert to mesh
    mesh = point_cloud_to_mesh(pcd, method=method)
    
    # Transfer colors from point cloud to mesh vertices
    if pcd.has_colors():
        # Sample colors from point cloud to mesh vertices
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[:len(mesh.vertices)]
        )
    
    return mesh


def save_mesh(mesh, output_path, format='ply'):
    """
    Save mesh to file
    
    Args:
        mesh: o3d.geometry.TriangleMesh
        output_path: output file path
        format: 'ply', 'obj', or 'stl'
    """
    if format == 'ply':
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
    elif format == 'obj':
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=False)
    elif format == 'stl':
        o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"💾 Mesh saved to: {output_path}")


def visualize_mesh(mesh, window_name="3D Mesh"):
    """Visualize mesh in Open3D viewer"""
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        width=1280,
        height=720,
        mesh_show_back_face=True
    )
