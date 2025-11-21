import os
import open3d as o3d
import numpy as np

def find_mesh_or_pointcloud(dense_dir):
    """Find mesh or point cloud file in dense directory
    
    Args:
        dense_dir: Path to dense reconstruction directory (or inference results)
        
    Returns:
        tuple: (file_path, file_type) where file_type is 'mesh' or 'pointcloud'
    """
    # Priority 0: Check for inference results (mesh_3d.ply/obj in root)
    mesh_3d_ply = os.path.join(dense_dir, "mesh_3d.ply")
    if os.path.exists(mesh_3d_ply):
        return mesh_3d_ply, 'mesh'
    
    mesh_3d_obj = os.path.join(dense_dir, "mesh_3d.obj")
    if os.path.exists(mesh_3d_obj):
        return mesh_3d_obj, 'mesh'
    
    # Priority 1: Check mesh directory for surface meshes (usually has best vertex colors)
    mesh_dir = os.path.join(dense_dir, "mesh")
    if os.path.exists(mesh_dir):
        # Prefer surface_mesh (larger, more colors) over refined_mesh
        surface = os.path.join(mesh_dir, "surface_mesh.ply")
        if os.path.exists(surface):
            return surface, 'mesh'
        
        refined = os.path.join(mesh_dir, "refined_mesh.ply")
        if os.path.exists(refined):
            return refined, 'mesh'
        
        # Fallback to any mesh file
        for filename in os.listdir(mesh_dir):
            if filename.endswith(('.ply', '.obj', '.off')):
                return os.path.join(mesh_dir, filename), 'mesh'
    
    # Priority 2: Check for textured mesh in root dense directory
    textured_ply = os.path.join(dense_dir, "textured_mesh.ply")
    if os.path.exists(textured_ply):
        return textured_ply, 'mesh'
    
    textured_obj = os.path.join(dense_dir, "textured_mesh.obj")
    if os.path.exists(textured_obj):
        return textured_obj, 'mesh'
    
    # Priority 3: Check dense/dense subdirectory (COLMAP output)
    dense_subdir = os.path.join(dense_dir, "dense")
    if os.path.exists(dense_subdir):
        # Check for meshed point clouds first
        meshed_poisson = os.path.join(dense_subdir, "meshed-poisson.ply")
        if os.path.exists(meshed_poisson):
            return meshed_poisson, 'mesh'
        
        meshed_delaunay = os.path.join(dense_subdir, "meshed-delaunay.ply")
        if os.path.exists(meshed_delaunay):
            return meshed_delaunay, 'mesh'
        
        # Then check for point clouds
        fused = os.path.join(dense_subdir, "fused.ply")
        if os.path.exists(fused):
            return fused, 'pointcloud'
    
    # Priority 4: Check for point cloud in root dense directory
    fused_ply = os.path.join(dense_dir, "fused.ply")
    if os.path.exists(fused_ply):
        return fused_ply, 'pointcloud'
    
    dense_ply = os.path.join(dense_dir, "dense.ply")
    if os.path.exists(dense_ply):
        return dense_ply, 'pointcloud'
    
    return None, None

def setup_visualizer(window_name="3D Reconstruction Viewer"):
    """Setup Open3D visualizer with custom settings
    
    Args:
        window_name: Name of visualization window
        
    Returns:
        o3d.visualization.Visualizer: Configured visualizer
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=1280,
        height=720,
        left=50,
        top=50
    )
    
    # Get render options
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.05, 0.05, 0.05])  # Almost black
    render_option.point_size = 2.0
    render_option.line_width = 1.0
    render_option.show_coordinate_frame = True
    render_option.mesh_show_back_face = True
    render_option.mesh_show_wireframe = False
    
    # Enable better lighting
    render_option.light_on = True
    render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
    
    return vis

def visualize_outputs(dense_dir):
    """Visualize 3D reconstruction outputs (mesh or point cloud)
    
    Args:
        dense_dir: Path to dense reconstruction directory
    """
    print("\n" + "="*60)
    print("3D RECONSTRUCTION VISUALIZATION")
    print("="*60)
    print(f"📂 Directory: {dense_dir}\n")
    
    # Check if directory exists
    if not os.path.exists(dense_dir):
        print(f"❌ Error: Directory not found: {dense_dir}")
        return
    
    # Find mesh or point cloud
    file_path, file_type = find_mesh_or_pointcloud(dense_dir)
    
    if file_path is None:
        print("❌ No mesh or point cloud found in directory!")
        print("\nExpected files (in priority order):")
        print("  1. dense/mesh/surface_mesh.ply (BEST - vertex colors)")
        print("  2. dense/mesh/refined_mesh.ply")
        print("  3. dense/textured_mesh.ply or .obj")
        print("  4. dense/dense/meshed-poisson.ply")
        print("  5. dense/dense/fused.ply (point cloud)")
        print("  6. dense/fused.ply or dense.ply")
        print(f"\nActual directory contents:")
        if os.path.exists(dense_dir):
            for item in os.listdir(dense_dir):
                item_path = os.path.join(dense_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        return
    
    print(f"✅ Found {file_type}: {os.path.basename(file_path)}")
    
    # Load geometry based on type
    try:
        if file_type == 'mesh':
            print(f"📥 Loading mesh...")
            geometry = o3d.io.read_triangle_mesh(file_path)
            
            # Check if texture file exists for .obj
            if file_path.endswith('.obj'):
                mtl_path = file_path.replace('.obj', '.mtl')
                if os.path.exists(mtl_path):
                    print(f"✅ Found texture file: {os.path.basename(mtl_path)}")
            
            # Compute normals if not present
            if not geometry.has_vertex_normals():
                print("🔧 Computing vertex normals...")
                geometry.compute_vertex_normals()
            
            # Get mesh statistics
            n_vertices = len(geometry.vertices)
            n_triangles = len(geometry.triangles)
            has_colors = geometry.has_vertex_colors()
            has_textures = geometry.has_textures()
            
            print(f"\n📊 Mesh Statistics:")
            print(f"   Vertices:  {n_vertices:,}")
            print(f"   Triangles: {n_triangles:,}")
            print(f"   Colors:    {'Yes' if has_colors else 'No'}")
            print(f"   Textures:  {'Yes' if has_textures else 'No'}")
            
            # Paint mesh with uniform color if no vertex colors
            if not has_colors:
                print("🎨 No vertex colors found, applying gray material...")
                geometry.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
            
        else:  # pointcloud
            print(f"📥 Loading point cloud...")
            geometry = o3d.io.read_point_cloud(file_path)
            
            # Get point cloud statistics
            n_points = len(geometry.points)
            has_colors = geometry.has_colors()
            has_normals = geometry.has_normals()
            
            print(f"\n📊 Point Cloud Statistics:")
            print(f"   Points:  {n_points:,}")
            print(f"   Colors:  {'Yes' if has_colors else 'No'}")
            print(f"   Normals: {'Yes' if has_normals else 'No'}")
            
            # Estimate normals if not present
            if not has_normals:
                print("🔧 Estimating normals...")
                geometry.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
        
        # Create visualizer
        print("\n🎨 Opening 3D viewer...")
        vis = setup_visualizer(f"3D Viewer - {os.path.basename(file_path)}")
        
        # Add geometry
        vis.add_geometry(geometry)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)
        
        # Print controls
        print("\n⌨️  Viewer Controls:")
        print("   Left Mouse:   Rotate")
        print("   Right Mouse:  Pan")
        print("   Scroll:       Zoom")
        print("   H:            Show help")
        print("   Q/ESC:        Quit")
        print("   +/-:          Adjust point size")
        print("   Ctrl+C:       Copy camera parameters")
        
        # Set nice initial view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        
        # Run visualizer
        vis.run()
        vis.destroy_window()
        
        print("\n✅ Visualization closed")
        
    except Exception as e:
        print(f"\n❌ Error loading geometry: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_comparison(original_dir, restored_dir, artifact_name="Artifact"):
    """Visualize original vs restored reconstruction side-by-side
    
    Args:
        original_dir: Path to original dense reconstruction
        restored_dir: Path to restored dense reconstruction  
        artifact_name: Name of artifact for display
    """
    print("\n" + "="*60)
    print(f"COMPARISON VIEW: {artifact_name}")
    print("="*60 + "\n")
    
    geometries = []
    labels = []
    
    # Load original
    orig_file, orig_type = find_mesh_or_pointcloud(original_dir)
    if orig_file:
        print(f"📥 Loading original {orig_type}...")
        if orig_type == 'mesh':
            orig_geom = o3d.io.read_triangle_mesh(orig_file)
            orig_geom.compute_vertex_normals()
        else:
            orig_geom = o3d.io.read_point_cloud(orig_file)
        
        # Move to left
        orig_geom.translate([-2, 0, 0])
        geometries.append(orig_geom)
        labels.append("Original")
        print(f"   ✅ Loaded original")
    
    # Load restored
    rest_file, rest_type = find_mesh_or_pointcloud(restored_dir)
    if rest_file:
        print(f"📥 Loading restored {rest_type}...")
        if rest_type == 'mesh':
            rest_geom = o3d.io.read_triangle_mesh(rest_file)
            rest_geom.compute_vertex_normals()
        else:
            rest_geom = o3d.io.read_point_cloud(rest_file)
        
        # Move to right
        rest_geom.translate([2, 0, 0])
        geometries.append(rest_geom)
        labels.append("Restored")
        print(f"   ✅ Loaded restored")
    
    if not geometries:
        print("❌ No geometries to compare!")
        return
    
    # Add coordinate frames
    coord_left = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[-2, 0, 0]
    )
    coord_right = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[2, 0, 0]
    )
    geometries.extend([coord_left, coord_right])
    
    # Visualize
    print(f"\n🎨 Comparing {' vs '.join(labels)}...")
    print("   Left side:  Original")
    print("   Right side: Restored\n")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Comparison: {artifact_name}",
        width=1920,
        height=1080,
        left=50,
        top=50
    )
    
    print("✅ Comparison closed")

def save_screenshot(dense_dir, output_path="screenshot.png", resolution=(1920, 1080)):
    """Save screenshot of 3D reconstruction
    
    Args:
        dense_dir: Path to dense reconstruction directory
        output_path: Output path for screenshot
        resolution: Screenshot resolution (width, height)
    """
    file_path, file_type = find_mesh_or_pointcloud(dense_dir)
    
    if file_path is None:
        print("❌ No geometry found to screenshot")
        return False
    
    try:
        # Load geometry
        if file_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(file_path)
            geometry.compute_vertex_normals()
        else:
            geometry = o3d.io.read_point_cloud(file_path)
        
        # Create visualizer (off-screen)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=resolution[0], height=resolution[1])
        vis.add_geometry(geometry)
        
        # Set view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Capture
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        
        print(f"✅ Screenshot saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving screenshot: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize 3D reconstruction outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize inference results
  python src/visualization/visualize_outputs.py --results_dir inference_results/single/damaged_artifact
  
  # Visualize COLMAP reconstruction
  python src/visualization/visualize_outputs.py --results_dir outputs/boy_with_thorn/dense
  
  # Compare original vs restored
  python src/visualization/visualize_outputs.py --compare --original outputs/artifact/dense --restored outputs/artifact/reconstruction_from_restored/dense
  
  # Save screenshot
  python src/visualization/visualize_outputs.py --results_dir inference_results/single/artifact --screenshot output.png
        """
    )
    
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to results directory (inference results or dense reconstruction)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Enable comparison mode (requires --original and --restored)"
    )
    parser.add_argument(
        "--original", type=str,
        help="Path to original reconstruction (for comparison mode)"
    )
    parser.add_argument(
        "--restored", type=str,
        help="Path to restored reconstruction (for comparison mode)"
    )
    parser.add_argument(
        "--screenshot", type=str,
        help="Save screenshot to specified path instead of opening viewer"
    )
    parser.add_argument(
        "--artifact_name", type=str, default="Artifact",
        help="Artifact name for comparison display"
    )
    
    args = parser.parse_args()
    
    # Comparison mode
    if args.compare:
        if not args.original or not args.restored:
            parser.error("--compare requires both --original and --restored")
        visualize_comparison(args.original, args.restored, args.artifact_name)
    
    # Screenshot mode
    elif args.screenshot:
        save_screenshot(args.results_dir, args.screenshot)
    
    # Normal visualization mode
    else:
        visualize_outputs(args.results_dir)
