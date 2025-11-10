import argparse
import os
import torch
from src.recons_pipeline import run_pipeline
from src.datasets.dataset import get_subdatasets, initial_done
from src.sfm.sfm_pycolmap import run_sfm
from visualization import visualize_outputs
from src.ai.restoration_model import PretrainedRestorationModel
from src.ai.trainer import RestorationTrainer
from src.datasets.restoration_dataset import RestorationDataset
from src.texturing.texturing import run_texturing
from src.datasets.restoration_dataset import restore_images_with_model
from src.ai.single_inference import run_single_inference
from src.ai.multi_inference import run_multi_inference
from src.ai.self_supervised import run_self_supervised_pretraining


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_mesh_path(dense_dir):
    """Find mesh file (.ply or .obj) in dense/mesh directory"""
    mesh_dir = os.path.join(dense_dir, "mesh")
    if not os.path.isdir(mesh_dir):
        return None
    for fn in os.listdir(mesh_dir):
        if fn.lower().endswith(".ply") or fn.lower().endswith(".obj"):
            return os.path.join(mesh_dir, fn)
    return None


def get_device():
    """Get available device (CUDA or CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    return device


def load_model(checkpoint_dir, device):
    """Load trained model from checkpoint"""
    best_model = os.path.join(checkpoint_dir, "best_model.pth")
    last_model = os.path.join(checkpoint_dir, "last_model.pth")
    
    # Prefer last_model, fallback to best_model
    model_path = last_model if os.path.exists(last_model) else best_model
    model_name = "Last Model" if os.path.exists(last_model) else "Best Model"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"❌ No model found at {checkpoint_dir}")
    
    print(f"✅ Loading {model_name} from {model_path}")
    model = PretrainedRestorationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def collect_dataset_pairs(output_root, artifact_name=None):
    """Collect training/validation dataset pairs from outputs folder"""
    targets = [artifact_name] if artifact_name else [
        d for d in os.listdir(output_root) 
        if os.path.isdir(os.path.join(output_root, d))
    ]
    
    train_pairs = []
    val_pairs = []
    
    for t in targets:
        base = os.path.join(output_root, t)
        images_dir = os.path.join(base, "images")
        depth_dir = os.path.join(base, "dense", "depth_maps")
        
        # Check if both directories exist
        if not (os.path.isdir(images_dir) and os.path.isdir(depth_dir)):
            continue
        
        # Split: 80% train, 20% validation (based on hash)
        if hash(t) % 5 == 0:
            val_pairs.append((images_dir, depth_dir))
        else:
            train_pairs.append((images_dir, depth_dir))
    
    return train_pairs, val_pairs



# ============================================================================
# MODE: RECONSTRUCT
# ============================================================================

def mode_reconstruct(args, output_root):
    """Run 3D reconstruction pipeline on input datasets"""
    print("\n" + "="*60)
    print("MODE: RECONSTRUCT - 3D Reconstruction Pipeline")
    print("="*60 + "\n")
    
    # Initialize done tracking
    done_outputs = None if args.skip_done == "True" else initial_done(output_root)
    
    # Get datasets to process
    datasets = get_subdatasets(
        args.dataset, 
        output_root=output_root, 
        is_augment=True, 
        done_outputs=done_outputs
    )
    
    # Process each dataset
    for name, paths in datasets.items():
        print(f"\n{'='*60}")
        print(f"🚀 Processing dataset: {name}")
        print(f"{'='*60}")
        
        # Step 1: Structure from Motion
        print("\n📸 Step 1: Running SfM...")
        maps = run_sfm(paths)
        if not maps:
            print(f"❌ SfM failed for {name}, skipping...")
            continue
        
        # Step 2: Full reconstruction pipeline (MVS + Mesh)
        print("\n🏗️ Step 2: Running reconstruction pipeline...")
        run_pipeline(paths, output_dir=paths["dense"])
        
        # Mark as done
        with open(f'{output_root}/done.txt', 'a', encoding='utf-8') as f:
            f.write(name + ',')
        
        print(f"\n✅ {name} completed!")
    
    print("\n" + "="*60)
    print("RECONSTRUCT MODE COMPLETED")
    print("="*60 + "\n")


# ============================================================================
# MODE: TRAIN
# ============================================================================

def mode_train(args, output_root, device):
    """Train restoration model using reconstructed data"""
    print("\n" + "="*60)
    print("MODE: TRAIN - Model Training")
    print("="*60 + "\n")
    
    # Collect dataset pairs
    train_pairs, val_pairs = collect_dataset_pairs(output_root, args.artifact)
    
    if not train_pairs:
        raise RuntimeError(
            "❌ No training data found!\n"
            "Make sure outputs/*/images and outputs/*/dense/depth_maps exist.\n"
            "Run --mode reconstruct first."
        )
    
    print(f"📊 Dataset statistics:")
    print(f"   Training pairs: {len(train_pairs)}")
    print(f"   Validation pairs: {len(val_pairs)}")
    
    # Create datasets
    train_ds = RestorationDataset(pairs=train_pairs, augment=True, use_advanced_damage=True)
    val_ds = RestorationDataset(pairs=val_pairs, augment=False, use_advanced_damage=False) if val_pairs else None
    
    print(f"\n📦 Loaded {len(train_ds)} training samples")
    if val_ds:
        print(f"📦 Loaded {len(val_ds)} validation samples")
    
    # Initialize model and trainer
    # Check if we should load pretrained encoder
    pretrained_encoder_path = None
    if args.use_pretrained_encoder:
        pretrained_path = "checkpoints/pretrained_encoder.pth"
        if os.path.exists(pretrained_path):
            pretrained_encoder_path = pretrained_path
            print(f"✅ Will use pretrained encoder from {pretrained_path}")
        else:
            print(f"⚠️ Pretrained encoder not found at {pretrained_path}, using ImageNet weights")
    
    model = PretrainedRestorationModel(load_pretrained_encoder=pretrained_encoder_path).to(device)
    trainer = RestorationTrainer(
        model=model,
        device=device,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoints_dir="checkpoints"
    )
    
    # Train
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    trainer.train(epochs=args.epochs, batch_size=args.batch_size)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60 + "\n")


# ============================================================================
# MODE: SINGLE INFERENCE (NEW!)
# ============================================================================

def mode_single_inference(args, device):
    """
    Inference for single damaged image
    Input: 1 damaged image
    Output: Restored 2D image + 3D mesh
    """
    print("\n" + "="*60)
    print("MODE: SINGLE INFERENCE - Single Image Restoration")
    print("="*60 + "\n")
    
    # Validate input
    if not os.path.exists(args.input_image):
        raise FileNotFoundError(f"❌ Input image not found: {args.input_image}")
    
    # Load model
    model = load_model("checkpoints", device)
    
    # Setup output directory
    if args.output_dir is None:
        base_name = os.path.splitext(os.path.basename(args.input_image))[0]
        output_dir = os.path.join("inference_results", "single", base_name)
    else:
        output_dir = args.output_dir
    
    # Run pipeline
    results = run_single_inference(
        model=model,
        device=device,
        input_image=args.input_image,
        output_dir=output_dir,
        visualize=not args.no_viz
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SINGLE INFERENCE COMPLETED")
    print("="*60)
    print(f"📁 Results saved to: {output_dir}")
    print(f"  - Restored image: {results['restored_image']}")
    print(f"  - Depth map: {results['depth_map']}")
    print(f"  - 3D mesh: {', '.join(results['mesh_files'])}")
    print("="*60 + "\n")


# ============================================================================
# MODE: MULTI INFERENCE (NEW!)
# ============================================================================

def mode_multi_inference(args, device):
    """
    Inference for multiple damaged images
    Input: Folder with damaged images
    Output: Model 3D with restored textures
    """
    print("\n" + "="*60)
    print("MODE: MULTI INFERENCE - Multi-View Restoration")
    print("="*60 + "\n")
    
    # Validate input
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"❌ Input directory not found: {args.input_dir}")
    
    # Load model
    model = load_model("checkpoints", device)
    
    # Setup output directory
    if args.output_dir is None:
        dir_name = os.path.basename(args.input_dir.rstrip('/\\'))
        output_dir = os.path.join("inference_results", "multi", dir_name)
    else:
        output_dir = args.output_dir
    
    # Run pipeline
    results = run_multi_inference(
        model=model,
        device=device,
        input_dir=args.input_dir,
        output_dir=output_dir,
        estimate_depth=args.estimate_depth,
        visualize=not args.no_viz
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MULTI INFERENCE COMPLETED")
    print("="*60)
    print(f"📁 Results saved to: {output_dir}")
    print(f"  - Restored images: {results['restored_images']}")
    if results['depth_maps']:
        print(f"  - Depth maps: {results['depth_maps']}")
    if results['dense']:
        print(f"  - 3D reconstruction: {results['dense']}")
    print("="*60 + "\n")


# ============================================================================
# MODE: PRETRAIN (NEW!)
# ============================================================================

def mode_pretrain(args, device):
    """
    Self-supervised pretraining on unlabeled images
    Helps model learn better representations before fine-tuning
    """
    print("\n" + "="*60)
    print("MODE: SELF-SUPERVISED PRETRAINING")
    print("="*60 + "\n")
    
    # Validate input
    if not os.path.isdir(args.pretrain_data):
        raise NotADirectoryError(f"❌ Pretrain data directory not found: {args.pretrain_data}")
    
    # Create model
    print("🔧 Initializing model for pretraining...")
    model = PretrainedRestorationModel().to(device)
    
    # Run self-supervised pretraining
    run_self_supervised_pretraining(
        model=model,
        device=device,
        unlabeled_data_dir=args.pretrain_data,
        task=args.pretrain_task,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        checkpoints_dir="checkpoints"
    )
    
    print("\n" + "="*60)
    print("✅ PRETRAINING COMPLETED")
    print("="*60)
    print(f"Pretrained encoder saved to: checkpoints/pretrained_encoder.pth")
    print("\nNext steps:")
    print("  1. Use pretrained encoder in training:")
    print("     python main.py --mode train --use_pretrained_encoder")
    print("  2. Or continue with regular training:")
    print("     python main.py --mode train")
    print("="*60 + "\n")


def restore_and_reconstruct(artifact_name, base_dir, model, device, batch_size):
    """Restore images and reconstruct 3D from restored images"""
    print(f"\n{'='*60}")
    print(f"Processing: {artifact_name}")
    print(f"{'='*60}")
    
    # Setup paths
    images_dir = os.path.join(base_dir, "images")
    depth_dir = os.path.join(base_dir, "dense", "depth_maps")
    restored_dir = os.path.join(base_dir, "restored_images")
    reconstruction_dir = os.path.join(base_dir, "reconstruction_from_restored")
    
    # Validate input directories
    if not (os.path.isdir(images_dir) and os.path.isdir(depth_dir)):
        print(f"⚠️ Skipping {artifact_name}: missing images or depth_maps")
        return None
    
    # Step 1: Restore images
    print("\n🎨 Step 1: Restoring images with trained model...")
    os.makedirs(restored_dir, exist_ok=True)
    metrics = restore_images_with_model(
        model, device, images_dir, depth_dir,
        out_dir=restored_dir, batch_size=batch_size
    )
    
    # Step 2: Reconstruct 3D from restored images
    print("\n🏗️ Step 2: Reconstructing 3D from restored images...")
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    recons_paths = {
        "images": restored_dir,
        "sparse": os.path.join(reconstruction_dir, "sparse"),
        "dense": os.path.join(reconstruction_dir, "dense")
    }
    
    # Run SfM
    maps = run_sfm(recons_paths)
    if not maps:
        print(f"❌ SfM failed for restored images of {artifact_name}")
        return metrics
    
    # Run full reconstruction
    run_pipeline(recons_paths, output_dir=recons_paths["dense"])
    
    # Step 3: Visualize results
    print("\n📊 Step 3: Visualizing reconstruction...")
    try:
        visualize_outputs(recons_paths["dense"])
    except Exception as e:
        print(f"⚠️ Visualization error: {str(e)}")
    
    return metrics


def mode_inference(args, output_root, device):
    """Run inference: restore images and reconstruct 3D"""
    print("\n" + "="*60)
    print("MODE: INFERENCE - Image Restoration & 3D Reconstruction")
    print("="*60 + "\n")
    
    # Load trained model
    model = load_model("checkpoints", device)
    
    # Get artifacts to process
    targets = [args.artifact] if args.artifact else [
        d for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ]
    
    print(f"📋 Processing {len(targets)} artifact(s): {', '.join(targets)}\n")
    
    # Process each artifact
    results = {}
    for artifact in targets:
        base_dir = os.path.join(output_root, artifact)
        metrics = restore_and_reconstruct(
            artifact, base_dir, model, device, args.batch_size
        )
        if metrics:
            results[artifact] = metrics
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE RESULTS SUMMARY")
    print("="*60)
    for artifact, metrics in results.items():
        print(f"  {artifact:30s} SSIM: {metrics['ssim']:.4f}")
    print("="*60 + "\n")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point - parse arguments and route to appropriate mode"""
    parser = argparse.ArgumentParser(
        description="3D Reconstruction + AI-based Image Restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Reconstruct 3D from images
  python main.py --mode reconstruct --dataset data/data_demo

  # Train restoration model
  python main.py --mode train --epochs 20 --batch_size 4

  # Single image inference
  python main.py --mode single_inference --input_image damaged.jpg

  # Multi-view inference
  python main.py --mode multi_inference --input_dir damaged_images/

  # Legacy inference (from outputs folder)
  python main.py --mode inference --artifact boy_with_thorn
        """
    )
    
    parser.add_argument(
        "--dataset", type=str, default="data/data_demo",
        help="Path to input dataset folder (for reconstruct mode)"
    )
    parser.add_argument(
        "--mode", type=str, 
        choices=['reconstruct', 'train', 'inference', 'single_inference', 'multi_inference', 'pretrain'],
        default="reconstruct",
        help="Operation mode"
    )
    parser.add_argument(
        "--skip_done", type=str, 
        choices=['True', 'False'], 
        default="True",
        help="Skip already processed datasets in reconstruct mode"
    )
    parser.add_argument(
        "--artifact", type=str, default=None,
        help="Specific artifact name (for train/inference mode, default=all)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for training/inference"
    )
    
    # NEW: Single inference arguments
    parser.add_argument(
        "--input_image", type=str, default=None,
        help="Input image path for single_inference mode"
    )
    
    # NEW: Multi inference arguments
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Input directory for multi_inference mode"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--estimate_depth", action="store_true",
        help="Estimate depth maps for multi_inference"
    )
    parser.add_argument(
        "--no_viz", action="store_true",
        help="Skip 3D visualization"
    )
    
    # NEW: Pretrain arguments
    parser.add_argument(
        "--pretrain_data", type=str, default=None,
        help="Directory with unlabeled images for self-supervised pretraining"
    )
    parser.add_argument(
        "--pretrain_task", type=str, default="rotation",
        choices=['rotation', 'jigsaw'],
        help="Self-supervised task for pretraining"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10,
        help="Number of pretraining epochs"
    )
    parser.add_argument(
        "--use_pretrained_encoder", action="store_true",
        help="Use pretrained encoder from self-supervised pretraining in training mode"
    )
    
    args = parser.parse_args()
    
    # Setup
    output_root = "outputs"
    os.makedirs(output_root, exist_ok=True)
    os.makedirs("inference_results", exist_ok=True)
    device = get_device()
    
    # Route to appropriate mode
    if args.mode == "reconstruct":
        mode_reconstruct(args, output_root)
    
    elif args.mode == "train":
        mode_train(args, output_root, device)
    
    elif args.mode == "inference":
        mode_inference(args, output_root, device)
    
    elif args.mode == "single_inference":
        if args.input_image is None:
            parser.error("--input_image is required for single_inference mode")
        mode_single_inference(args, device)
    
    elif args.mode == "multi_inference":
        if args.input_dir is None:
            parser.error("--input_dir is required for multi_inference mode")
        mode_multi_inference(args, device)
    
    elif args.mode == "pretrain":
        if args.pretrain_data is None:
            parser.error("--pretrain_data is required for pretrain mode")
        mode_pretrain(args, device)


if __name__ == "__main__":
    main()