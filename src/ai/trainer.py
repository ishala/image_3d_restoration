import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import VGG16_Weights


class MultiLayerPerceptualLoss(nn.Module):
    """Enhanced VGG-based perceptual loss using multiple layers for richer feature matching"""
    def __init__(self):
        super().__init__()
        # Load VGG16 with new API
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract multiple layers for richer feature matching
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2 (64 channels)
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2 (128 channels)
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # relu3_3 (256 channels)
        self.slice4 = nn.Sequential(*list(vgg[16:23]))# relu4_3 (512 channels)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Eval mode
        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()
        self.slice4.eval()
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Layer weights (can be tuned)
        self.layer_weights = [1.0, 1.0, 1.0, 1.0]  # Equal weight for all layers

    def normalize_for_vgg(self, x):
        """Normalize input for VGG network"""
        # Denormalize from [-1,1] to [0,1]
        x = (x + 1) / 2
        # Normalize for ImageNet
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        # Normalize inputs
        pred = self.normalize_for_vgg(pred)
        target = self.normalize_for_vgg(target)
        
        # Compute multi-layer perceptual loss
        loss = 0.0
        
        # Layer 1
        pred_1 = self.slice1(pred)
        target_1 = self.slice1(target)
        loss += self.layer_weights[0] * F.l1_loss(pred_1, target_1)
        
        # Layer 2
        pred_2 = self.slice2(pred_1)
        target_2 = self.slice2(target_1)
        loss += self.layer_weights[1] * F.l1_loss(pred_2, target_2)
        
        # Layer 3
        pred_3 = self.slice3(pred_2)
        target_3 = self.slice3(target_2)
        loss += self.layer_weights[2] * F.l1_loss(pred_3, target_3)
        
        # Layer 4
        pred_4 = self.slice4(pred_3)
        target_4 = self.slice4(target_3)
        loss += self.layer_weights[3] * F.l1_loss(pred_4, target_4)
        
        # Average across layers
        return loss / sum(self.layer_weights)


# Keep old version for backward compatibility
class PerceptualLoss(MultiLayerPerceptualLoss):
    """Alias for backward compatibility"""
    pass


class RestorationTrainer:
    def __init__(self, model, device, train_dataset, val_dataset=None, checkpoints_dir="checkpoints"):
        self.model = model
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Separate param groups with different learning rates
        encoder_params = []
        decoder_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'enc' in n or 'encoder' in n:
                encoder_params.append(p)
            else:
                decoder_params.append(p)
        
        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": 1e-5})  # Lower LR for pretrained
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": 5e-5})  # Conservative LR
        if not param_groups:
            param_groups = [{"params": self.model.parameters(), "lr": 5e-5}]
        
        self.optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        
        # Enhanced loss: L1 + Multi-layer Perceptual + SSIM
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = MultiLayerPerceptualLoss().to(device)
        
        # SSIM loss for structure preservation
        try:
            from pytorch_msssim import SSIM
            self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
            self.use_ssim = True
            print("✅ SSIM loss enabled")
        except ImportError:
            self.ssim_loss = None
            self.use_ssim = False
            print("⚠️ pytorch-msssim not installed, SSIM loss disabled")
        
        print(f"✅ Trainer initialized - Encoder params: {len(encoder_params)}, Decoder params: {len(decoder_params)}")
        print(f"   Loss: L1 + Multi-layer Perceptual" + (" + SSIM" if self.use_ssim else ""))

    def _make_loader(self, dataset, batch_size=4, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    def train(self, epochs=20, batch_size=4):
        train_loader = self._make_loader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self._make_loader(self.val_dataset, batch_size=batch_size, shuffle=False) if self.val_dataset else None
        best_val = float("inf")

        for ep in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            n = 0
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    img = batch["damaged"].to(self.device)
                    depth = batch["depth"].to(self.device)
                    gt = batch["ground_truth"].to(self.device)
                    
                    # Validate inputs
                    if not (torch.isfinite(img).all() and torch.isfinite(depth).all() and torch.isfinite(gt).all()):
                        print(f"⚠️ Skipping batch {batch_idx} - non-finite values")
                        continue

                    # Forward pass
                    out = self.model(img, depth)
                    
                    # Check output
                    if not torch.isfinite(out).all():
                        print(f"⚠️ Non-finite output at batch {batch_idx}")
                        continue
                    
                    # Enhanced combined loss: L1 + Perceptual + SSIM
                    l1 = self.l1_loss(out, gt)
                    perceptual = self.perceptual_loss(out, gt)
                    
                    if self.use_ssim:
                        # Denormalize to [0, 1] for SSIM
                        out_01 = (out + 1) / 2
                        gt_01 = (gt + 1) / 2
                        ssim_val = self.ssim_loss(out_01, gt_01)
                        ssim_loss = 1 - ssim_val  # Convert to loss
                        
                        # Weighted combination: 50% L1 + 30% Perceptual + 20% SSIM
                        loss = 0.5 * l1 + 0.3 * perceptual + 0.2 * ssim_loss
                    else:
                        # Fallback: 70% L1 + 30% Perceptual
                        loss = 0.7 * l1 + 0.3 * perceptual
                    
                    if not torch.isfinite(loss):
                        print(f"⚠️ NaN loss at batch {batch_idx}")
                        continue

                    # Backward pass with gradient clipping
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    running += loss.item()
                    n += 1
                    
                    # Print progress
                    if batch_idx % 10 == 0:
                        print(f"Epoch {ep}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
                        
                except Exception as e:
                    print(f"❌ Error at batch {batch_idx}: {str(e)}")
                    continue

            train_loss = running / max(1, n)
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)

            print(f"\n{'='*50}")
            print(f"Epoch {ep}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.6f}")
                
                # Save best model
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "best_model.pth"))
                    print(f"✅ Best model saved (val loss: {val_loss:.6f})")
            print(f"{'='*50}\n")

        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.checkpoints_dir, "last_model.pth"))
        print(f"✅ Training completed. Final model saved.")

    def validate(self, val_loader):
        self.model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    img = batch["damaged"].to(self.device)
                    depth = batch["depth"].to(self.device)
                    gt = batch["ground_truth"].to(self.device)
                    
                    # Validate inputs
                    if not (torch.isfinite(img).all() and torch.isfinite(depth).all() and torch.isfinite(gt).all()):
                        continue

                    out = self.model(img, depth)
                    
                    if not torch.isfinite(out).all():
                        continue
                    
                    # Enhanced combined loss
                    l1 = self.l1_loss(out, gt)
                    perceptual = self.perceptual_loss(out, gt)
                    
                    if self.use_ssim:
                        out_01 = (out + 1) / 2
                        gt_01 = (gt + 1) / 2
                        ssim_val = self.ssim_loss(out_01, gt_01)
                        ssim_loss = 1 - ssim_val
                        loss = 0.5 * l1 + 0.3 * perceptual + 0.2 * ssim_loss
                    else:
                        loss = 0.7 * l1 + 0.3 * perceptual
                    
                    if torch.isfinite(loss):
                        running += loss.item()
                        n += 1
                        
                except Exception:
                    continue
                    
        return running / max(1, n) if n > 0 else float('inf')