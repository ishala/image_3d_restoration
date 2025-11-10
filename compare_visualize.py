import argparse
import os
import open3d as o3d

def load_mesh(artifact, reconstruction_type='restored'):
    """Load mesh from artifact reconstruction"""
    if reconstruction_type == 'restored':
        base_path = f"outputs/{artifact}/reconstruction_from_restored/dense"
    else:
        base_path = f"outputs/{artifact}/dense"
    
    # Try to find mesh
    mesh_dir = os.path.join(base_path, "mesh")
    if os.path.exists(mesh_dir):
        for file in os.listdir(mesh_dir):
            if file.endswith(('.ply', '.obj')):
                mesh_path = os.path.join(mesh_dir, file)
                print(f"📂 Loading: {mesh_path}")
                return o3d.io.read_triangle_mesh(mesh_path)
    
    # Try dense point cloud
    dense_ply = os.path.join(base_path, "dense.ply")
    if os.path.exists(dense_ply):
        print(f"📂 Loading: {dense_ply}")
        pcd = o3d.io.read_point_cloud(dense_ply)
        return pcd
    
    print(f"❌ No mesh/point cloud found in {base_path}")
    return None

def compare_side_by_side(artifact):
    """Compare original vs restored reconstruction"""
    print("\n" + "="*60)
    print(f"Comparing reconstructions for: {artifact}")
    print("="*60 + "\n")
    
    # Load both meshes
    original = load_mesh(artifact, 'original')
    restored = load_mesh(artifact, 'restored')
    
    geometries = []
    
    # Position original on left
    if original is not None:
        if hasattr(original, 'compute_vertex_normals'):
            original.compute_vertex_normals()
        original.translate([-2, 0, 0])  # Move to left
        geometries.append(original)
        print("✅ Loaded original reconstruction")
    
    # Position restored on right
    if restored is not None:
        if hasattr(restored, 'compute_vertex_normals'):
            restored.compute_vertex_normals()
        restored.translate([2, 0, 0])  # Move to right
        geometries.append(restored)
        print("✅ Loaded restored reconstruction")
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    # Visualize
    if geometries:
        print("\n🎨 Opening 3D viewer...")
        print("Controls:")
        print("  - Mouse drag: Rotate")
        print("  - Scroll: Zoom")
        print("  - Right click drag: Pan")
        print("  - Press 'h': Help")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Comparison: {artifact} (Left=Original, Right=Restored)",
            width=1920,
            height=1080,
            left=50,
            top=50
        )
    else:
        print("❌ No geometries to visualize")

def main():
    parser = argparse.ArgumentParser(description="Compare original vs restored 3D reconstructions")
    parser.add_argument("--artifact", type=str, required=True,
                        help="Artifact name (e.g., boy_with_thorn)")
    args = parser.parse_args()
    
    compare_side_by_side(args.artifact)

if __name__ == "__main__":
    main()