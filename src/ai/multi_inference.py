"""
Multi-View Inference Pipeline
Process multiple damaged images to accurate 3D reconstruction
"""
import os
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.sfm.sfm_pycolmap import run_sfm
from src.recons_pipeline import run_pipeline
from visualization import visualize_outputs


class MultiViewPipeline:
    """Pipeline for processing multiple damaged images"""
    
    def __init__(self, restoration_model, device='cuda'):
        """
        Args:
            restoration_model: Trained restoration model
            device: 'cuda' or 'cpu'
        """
        self.restoration_model = restoration_model
        self.device = device
        self.restoration_model.eval()
        
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
    
    def restore_images_batch(self, images_dir, depth_dir, output_dir, batch_size=4):
        """
        Restore multiple images in batch
        
        Args:
            images_dir: directory with input images
            depth_dir: directory with estimated depth maps
            output_dir: directory to save restored images
            batch_size: batch size for processing
        
        Returns:
            output_dir path
        """
        print(f"🎨 Restoring images from {images_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect images
        image_files = sorted([f for f in os.listdir(images_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            imgs = []
            depths = []
            names = []
            
            for fname in batch_files:
                # Load image
                img_path = os.path.join(images_dir, fname)
                img = Image.open(img_path).convert('RGB')
                imgs.append(self.transform(img))
                
                # Load or estimate depth
                base = os.path.splitext(fname)[0]
                depth_path = os.path.join(depth_dir, f"{base}_depth.png")
                
                if os.path.exists(depth_path):
                    depth = Image.open(depth_path).convert('L')
                    depths.append(self.transform(depth))
                else:
                    # Use dummy depth if not available
                    depths.append(torch.ones_like(imgs[-1][:1]))
                
                names.append(fname)
            
            # Stack and process
            imgs_t = torch.stack(imgs).to(self.device)
            depths_t = torch.stack(depths).to(self.device)
            
            # Normalize
            imgs_t = imgs_t * 2.0 - 1.0
            depths_t = depths_t * 2.0 - 1.0
            
            # Restore
            with torch.no_grad():
                restored_t = self.restoration_model(imgs_t, depths_t)
            
            # Denormalize
            restored_t = (restored_t.clamp(-1, 1) + 1.0) / 2.0
            
            # Save
            for j, name in enumerate(names):
                out_path = os.path.join(output_dir, name)
                TF.to_pil_image(restored_t[j].cpu()).save(out_path, quality=95)
                print(f"  ✓ {name}")
        
        print(f"✅ Restored {len(image_files)} images")
        return output_dir
    
    def reconstruct_3d(self, images_dir, sparse_dir, dense_dir):
        """
        Run full 3D reconstruction pipeline
        
        Args:
            images_dir: directory with images
            sparse_dir: directory for sparse reconstruction
            dense_dir: directory for dense reconstruction
        
        Returns:
            success: bool
        """
        print("🏗️ Running 3D reconstruction...")
        
        paths = {
            'images': images_dir,
            'sparse': sparse_dir,
            'dense': dense_dir
        }
        
        # Ensure directories exist
        os.makedirs(sparse_dir, exist_ok=True)
        os.makedirs(dense_dir, exist_ok=True)
        
        # Step 1: SfM
        print("  Step 1/2: Structure from Motion...")
        maps = run_sfm(paths)
        
        if not maps:
            print("❌ SfM failed")
            return False
        
        # Step 2: MVS + Mesh
        print("  Step 2/2: Multi-View Stereo & Meshing...")
        run_pipeline(paths, output_dir=dense_dir)
        
        print("✅ 3D reconstruction completed")
        return True
    
    def process(self, input_images_dir, output_base_dir, estimate_depth=False, visualize=True):
        """
        Full multi-view pipeline
        
        Args:
            input_images_dir: directory with damaged input images
            output_base_dir: base directory for all outputs
            estimate_depth: estimate depth maps if not available
            visualize: show 3D visualization
        
        Returns:
            dict with paths to outputs
        """
        print("\n" + "="*60)
        print("🚀 MULTI-VIEW INFERENCE PIPELINE")
        print("="*60)
        print(f"Input: {input_images_dir}")
        print(f"Output: {output_base_dir}\n")
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Setup paths
        depth_dir = os.path.join(output_base_dir, "depth_maps")
        restored_dir = os.path.join(output_base_dir, "restored_images")
        sparse_dir = os.path.join(output_base_dir, "sparse")
        dense_dir = os.path.join(output_base_dir, "dense")
        
        # Step 1: Depth estimation (if needed)
        if estimate_depth:
            print("📏 Step 1: Estimating depth maps...")
            from src.ai.depth_estimation import run_depth_estimation
            run_depth_estimation(input_images_dir, depth_dir, device=self.device)
        else:
            # Use dummy depth directory
            os.makedirs(depth_dir, exist_ok=True)
            print("⏭️ Skipping depth estimation")
        
        # Step 2: Restore images
        print("\n🎨 Step 2: Restoring images...")
        self.restore_images_batch(input_images_dir, depth_dir, restored_dir)
        
        # Step 3: 3D Reconstruction
        print("\n🏗️ Step 3: 3D Reconstruction from restored images...")
        success = self.reconstruct_3d(restored_dir, sparse_dir, dense_dir)
        
        if not success:
            print("⚠️ 3D reconstruction failed, but restored images are saved")
        
        # Step 4: Visualize
        if visualize and success:
            print("\n📊 Step 4: Visualizing results...")
            try:
                visualize_outputs(dense_dir)
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
        
        # Summary
        print("\n" + "="*60)
        print("✅ MULTI-VIEW PIPELINE COMPLETED")
        print("="*60)
        
        results = {
            'input': input_images_dir,
            'depth_maps': depth_dir if estimate_depth else None,
            'restored_images': restored_dir,
            'sparse': sparse_dir if success else None,
            'dense': dense_dir if success else None,
            'output_dir': output_base_dir
        }
        
        return results


def run_multi_inference(model, device, input_dir, output_dir, estimate_depth=False, visualize=True):
    """
    Convenience function for multi-view inference
    
    Args:
        model: trained restoration model
        device: torch device
        input_dir: directory with input images
        output_dir: output directory
        estimate_depth: estimate depth maps
        visualize: show visualization
    
    Returns:
        dict with results
    """
    pipeline = MultiViewPipeline(model, device)
    return pipeline.process(input_dir, output_dir, estimate_depth, visualize)
