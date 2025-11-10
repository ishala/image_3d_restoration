"""
Single Image Inference Pipeline
Process single damaged image to restored 2D + 3D mesh
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.ai.depth_estimation import estimate_depth
from src.mesh.depth_to_mesh import create_textured_mesh, save_mesh, visualize_mesh


class SingleImagePipeline:
    """Pipeline for processing single damaged image"""
    
    def __init__(self, restoration_model, device='cuda'):
        """
        Args:
            restoration_model: Trained restoration model
            device: 'cuda' or 'cpu'
        """
        self.restoration_model = restoration_model
        self.device = device
        self.restoration_model.eval()
        
        # Transforms
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
    
    def estimate_depth_map(self, image_path, depth_model="Intel/dpt-hybrid-midas"):
        """
        Estimate depth from single image
        
        Args:
            image_path: path to input image
            depth_model: pretrained depth model name
        
        Returns:
            depth_map: numpy array [H, W]
        """
        print(f"📏 Estimating depth with {depth_model}...")
        depth_map = estimate_depth(image_path, model_name=depth_model, device=self.device)
        print(f"✅ Depth estimated: shape {depth_map.shape}")
        return depth_map
    
    def restore_image(self, image_path, depth_map):
        """
        Restore damaged image using trained model
        
        Args:
            image_path: path to damaged image
            depth_map: estimated depth map
        
        Returns:
            restored_image: PIL Image
        """
        print("🎨 Restoring image...")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        orig_size = img.size
        
        # Prepare inputs
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Resize depth to match image
        if depth_tensor.shape[-2:] != img_tensor.shape[-2:]:
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor, size=img_tensor.shape[-2:], mode='bilinear', align_corners=True
            )
        
        # Normalize to [-1, 1]
        img_tensor = img_tensor * 2.0 - 1.0
        depth_tensor = depth_tensor * 2.0 - 1.0
        
        # Restore
        with torch.no_grad():
            restored_tensor = self.restoration_model(img_tensor, depth_tensor)
        
        # Denormalize to [0, 1]
        restored_tensor = (restored_tensor.clamp(-1, 1) + 1.0) / 2.0
        
        # Convert to PIL
        restored_image = T.ToPILImage()(restored_tensor.squeeze(0).cpu())
        restored_image = restored_image.resize(orig_size, Image.LANCZOS)
        
        print("✅ Image restored")
        return restored_image
    
    def generate_3d_mesh(self, restored_image, depth_map, method='poisson'):
        """
        Generate 3D mesh from restored image and depth
        
        Args:
            restored_image: PIL Image
            depth_map: numpy array [H, W]
            method: reconstruction method
        
        Returns:
            mesh: o3d.geometry.TriangleMesh
        """
        print(f"🏗️ Generating 3D mesh using {method}...")
        
        # Resize depth to match image
        if depth_map.shape[:2] != (restored_image.height, restored_image.width):
            import cv2
            depth_map = cv2.resize(
                depth_map, 
                (restored_image.width, restored_image.height), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Create mesh
        mesh = create_textured_mesh(
            depth_map, 
            restored_image, 
            intrinsics=None,  # Auto-estimate
            method=method,
            depth_scale=5.0  # Adjust for better visualization
        )
        
        print("✅ 3D mesh generated")
        return mesh
    
    def process(self, input_image_path, output_dir, visualize=True, save_formats=['ply', 'obj']):
        """
        Full pipeline: input image → restored 2D + 3D mesh
        
        Args:
            input_image_path: path to damaged input image
            output_dir: directory to save results
            visualize: show 3D visualization
            save_formats: list of mesh formats to save
        
        Returns:
            dict with paths to all outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("🚀 SINGLE IMAGE INFERENCE PIPELINE")
        print("="*60)
        print(f"Input: {input_image_path}")
        print(f"Output: {output_dir}\n")
        
        # Step 1: Estimate depth
        depth_map = self.estimate_depth_map(input_image_path)
        
        # Save depth visualization
        depth_vis_path = os.path.join(output_dir, "depth_estimated.png")
        depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-9) * 255).astype(np.uint8)
        Image.fromarray(depth_normalized).save(depth_vis_path)
        print(f"💾 Depth map saved: {depth_vis_path}")
        
        # Step 2: Restore image
        restored_image = self.restore_image(input_image_path, depth_map)
        
        # Save restored image
        restored_path = os.path.join(output_dir, "restored.jpg")
        restored_image.save(restored_path, quality=95)
        print(f"💾 Restored image saved: {restored_path}")
        
        # Step 3: Generate 3D mesh
        mesh = self.generate_3d_mesh(restored_image, depth_map)
        
        # Save mesh in multiple formats
        mesh_paths = []
        for fmt in save_formats:
            mesh_path = os.path.join(output_dir, f"mesh_3d.{fmt}")
            save_mesh(mesh, mesh_path, format=fmt)
            mesh_paths.append(mesh_path)
        
        # Step 4: Visualize
        if visualize:
            print("\n📊 Launching 3D visualization...")
            visualize_mesh(mesh, window_name="Restored 3D Mesh")
        
        # Summary
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED")
        print("="*60)
        
        results = {
            'input': input_image_path,
            'depth_map': depth_vis_path,
            'restored_image': restored_path,
            'mesh_files': mesh_paths,
            'output_dir': output_dir
        }
        
        return results


def run_single_inference(model, device, input_image, output_dir, visualize=True):
    """
    Convenience function to run single image inference
    
    Args:
        model: trained restoration model
        device: torch device
        input_image: path to input image
        output_dir: output directory
        visualize: show visualization
    
    Returns:
        dict with results
    """
    pipeline = SingleImagePipeline(model, device)
    return pipeline.process(input_image, output_dir, visualize=visualize)
