"""
Comprehensive Evaluation Script for 3D Restoration Model
Evaluates 2D restoration quality and optional 3D reconstruction metrics
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import model and datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.ai.restoration_model import PretrainedRestorationModel
from src.datasets.restoration_dataset import RestorationDataset


class EvaluationMetrics:
    """Collection of evaluation metrics for image restoration"""
    
    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """Peak Signal-to-Noise Ratio"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def ssim(pred, target):
        """Structural Similarity Index (simplified version)"""
        try:
            from pytorch_msssim import ssim as pytorch_ssim
            # Convert from [-1,1] to [0,1]
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            return pytorch_ssim(pred_norm, target_norm, data_range=1.0, size_average=True)
        except ImportError:
            # Fallback to simple implementation
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            mu_x = F.avg_pool2d(pred, 3, 1, 1)
            mu_y = F.avg_pool2d(target, 3, 1, 1)
            
            sigma_x = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_x ** 2
            sigma_y = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2
            sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
            
            ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                       ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
            
            return ssim_map.mean()
    
    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return torch.mean(torch.abs(pred - target))
    
    @staticmethod
    def mse(pred, target):
        """Mean Squared Error"""
        return torch.mean((pred - target) ** 2)
    
    @staticmethod
    def lpips_score(pred, target, lpips_fn):
        """LPIPS perceptual similarity (requires lpips package)"""
        try:
            # Convert from [-1,1] to [0,1] for LPIPS
            pred_norm = (pred + 1) / 2
            target_norm = (target + 1) / 2
            return lpips_fn(pred_norm, target_norm).mean()
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            return torch.tensor(0.0)


class DamageDetectionMetrics:
    """Metrics for damage detection/segmentation evaluation"""
    
    @staticmethod
    def iou(pred_mask, target_mask, threshold=0.5):
        """Intersection over Union"""
        pred_binary = (pred_mask > threshold).float()
        target_binary = (target_mask > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return torch.tensor(1.0)
        return intersection / union
    
    @staticmethod
    def dice_score(pred_mask, target_mask, threshold=0.5):
        """Dice Coefficient"""
        pred_binary = (pred_mask > threshold).float()
        target_binary = (target_mask > threshold).float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum()
        
        if union == 0:
            return torch.tensor(1.0)
        return (2 * intersection) / union
    
    @staticmethod
    def pixel_accuracy(pred_mask, target_mask, threshold=0.5):
        """Pixel-wise accuracy"""
        pred_binary = (pred_mask > threshold).float()
        target_binary = (target_mask > threshold).float()
        
        correct = (pred_binary == target_binary).float().sum()
        total = pred_binary.numel()
        
        return correct / total


class ModelEvaluator:
    """Main evaluator class"""
    
    def __init__(self, model, device, use_lpips=False):
        self.model = model
        self.device = device
        self.metrics = EvaluationMetrics()
        self.damage_metrics = DamageDetectionMetrics()
        
        # Optional LPIPS
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
                self.lpips_fn.eval()
                print("✅ LPIPS loaded successfully")
            except ImportError:
                print("⚠️ LPIPS not available. Install with: pip install lpips")
    
    @torch.no_grad()
    def evaluate_batch(self, batch):
        """Evaluate a single batch"""
        damaged = batch['damaged'].to(self.device)
        clean = batch['clean'].to(self.device)
        depth = batch.get('depth', None)
        
        # Forward pass
        if depth is not None:
            depth = depth.to(self.device)
            restored = self.model(damaged, depth)
        else:
            restored = self.model(damaged)
        
        # Calculate metrics
        results = {
            'psnr': self.metrics.psnr(restored, clean).item(),
            'ssim': self.metrics.ssim(restored, clean).item(),
            'mae': self.metrics.mae(restored, clean).item(),
            'mse': self.metrics.mse(restored, clean).item(),
        }
        
        # LPIPS if available
        if self.lpips_fn is not None:
            results['lpips'] = self.metrics.lpips_score(restored, clean, self.lpips_fn).item()
        
        return results, restored
    
    def evaluate_dataset(self, dataset, batch_size=4, save_examples=True, output_dir=None):
        """Evaluate entire dataset"""
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING DATASET")
        print(f"{'='*70}\n")
        
        self.model.eval()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Collect results
        all_results = defaultdict(list)
        example_images = []
        
        # Evaluate batches
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                results, restored = self.evaluate_batch(batch)
                
                # Accumulate metrics
                for key, value in results.items():
                    all_results[key].append(value)
                
                # Save example images (first 5 batches)
                if save_examples and idx < 5 and output_dir:
                    self._save_example_batch(
                        batch, restored, idx, output_dir
                    )
        
        # Compute statistics
        stats = self._compute_statistics(all_results)
        
        return stats
    
    def _save_example_batch(self, batch, restored, batch_idx, output_dir):
        """Save example images for visualization"""
        examples_dir = os.path.join(output_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        
        # Get first image from batch
        damaged = batch['damaged'][0].cpu()
        clean = batch['clean'][0].cpu()
        restored_img = restored[0].cpu()
        
        # Denormalize from [-1,1] to [0,1]
        damaged = (damaged + 1) / 2
        clean = (clean + 1) / 2
        restored_img = (restored_img + 1) / 2
        
        # Create comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(damaged.permute(1, 2, 0).numpy())
        axes[0].set_title('Damaged Input')
        axes[0].axis('off')
        
        axes[1].imshow(restored_img.permute(1, 2, 0).numpy())
        axes[1].set_title('Restored')
        axes[1].axis('off')
        
        axes[2].imshow(clean.permute(1, 2, 0).numpy())
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(examples_dir, f"example_{batch_idx}.png"),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    def _compute_statistics(self, results):
        """Compute mean and std for all metrics"""
        stats = {}
        
        for metric_name, values in results.items():
            values_array = np.array(values)
            stats[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array))
            }
        
        return stats


def plot_metrics_comparison(results_dict, output_path):
    """Plot comparison of metrics across different models/settings"""
    metrics = ['psnr', 'ssim', 'mae']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[key][metric]['mean'] for key in results_dict.keys()]
        errors = [results_dict[key][metric]['std'] for key in results_dict.keys()]
        labels = list(results_dict.keys())
        
        axes[idx].bar(labels, values, yerr=errors, capsize=5)
        axes[idx].set_title(metric.upper())
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics comparison saved to: {output_path}")


def save_results_table(stats, output_path):
    """Save results as formatted table"""
    with open(output_path, 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write("| Metric | Mean | Std | Min | Max | Median |\n")
        f.write("|--------|------|-----|-----|-----|--------|\n")
        
        for metric, values in stats.items():
            f.write(f"| {metric.upper()} | "
                   f"{values['mean']:.4f} | "
                   f"{values['std']:.4f} | "
                   f"{values['min']:.4f} | "
                   f"{values['max']:.4f} | "
                   f"{values['median']:.4f} |\n")
    
    print(f"📄 Results table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D Restoration Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set
  python src/evaluation/evaluate.py --test_dir outputs/test_set --checkpoint checkpoints/best_model.pth
  
  # Evaluate with LPIPS
  python src/evaluation/evaluate.py --test_dir outputs/test_set --use_lpips
  
  # Compare multiple checkpoints
  python src/evaluation/evaluate.py --test_dir outputs/test_set --checkpoints checkpoints/*.pth
        """
    )
    
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--test_dir', type=str, required=True,
        help='Path to test dataset directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/eval',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--use_lpips', action='store_true',
        help='Use LPIPS perceptual metric (requires lpips package)'
    )
    parser.add_argument(
        '--save_examples', action='store_true', default=True,
        help='Save example comparisons'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("🔬 MODEL EVALUATION")
    print("="*70)
    print(f"📂 Test directory: {args.test_dir}")
    print(f"🎯 Checkpoint: {args.checkpoint}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔧 Device: {args.device}")
    print("="*70 + "\n")
    
    # Load model
    device = torch.device(args.device)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"📥 Loading model from {args.checkpoint}...")
    model = PretrainedRestorationModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Prepare test dataset
    # Assuming test_dir has structure: images/ and depth_maps/
    test_images = os.path.join(args.test_dir, "images")
    test_depth = os.path.join(args.test_dir, "dense", "depth_maps")
    
    if not os.path.exists(test_images):
        raise NotADirectoryError(f"Test images not found: {test_images}")
    
    print("📦 Loading test dataset...")
    test_dataset = RestorationDataset(
        pairs=[(test_images, test_depth)],
        augment=False,
        use_advanced_damage=False
    )
    print(f"✅ Loaded {len(test_dataset)} test samples\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, use_lpips=args.use_lpips)
    
    # Run evaluation
    stats = evaluator.evaluate_dataset(
        test_dataset,
        batch_size=args.batch_size,
        save_examples=args.save_examples,
        output_dir=args.output_dir
    )
    
    # Print results
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    for metric, values in stats.items():
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {values['mean']:.4f}")
        print(f"  Std:    {values['std']:.4f}")
        print(f"  Min:    {values['min']:.4f}")
        print(f"  Max:    {values['max']:.4f}")
        print(f"  Median: {values['median']:.4f}")
    print("="*70 + "\n")
    
    # Save results
    results_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Results saved to: {results_json}")
    
    # Save markdown table
    results_md = os.path.join(args.output_dir, "results_table.md")
    save_results_table(stats, results_md)
    
    print("\n✅ Evaluation completed!\n")


if __name__ == "__main__":
    main()
