import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ConvBlock(nn.Module):
    """Basic convolutional block for the decoder with GroupNorm"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),  # GroupNorm lebih stabil
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample + skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size() != skip.size():
            # Align spatial dims if necessary
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PretrainedRestorationModel(nn.Module):
    """
    ResUNet-style restoration model with pretrained ResNet50 encoder.
    Input: RGB (3) + Depth (1) → Output: Restored RGB (3)
    
    Features:
    - Pretrained ResNet50 encoder (ImageNet)
    - Optional: Load pretrained weights from image inpainting models
    - U-Net architecture with skip connections
    """
    def __init__(self, backbone='resnet50', unfreeze_last_layers=True, 
                 load_pretrained_encoder=None):
        super().__init__()

        # === Encoder (Pretrained ResNet50) ===
        if backbone == 'resnet50':
            print("🔹 Encoder backbone loaded:", backbone)
            encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)

            # Modify first conv layer to accept 4 channels
            orig_conv = encoder.conv1
            new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            new_conv.weight.data[:, :3, :, :] = orig_conv.weight.data
            new_conv.weight.data[:, 3:, :, :] = orig_conv.weight.data.mean(dim=1, keepdim=True)
            encoder.conv1 = new_conv
            self.encoder = encoder
        # === Extract Encoder Layers for Skip Connections ===
        self.enc1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)   # 64
        self.enc2 = nn.Sequential(self.encoder.maxpool, self.encoder.layer1)  # 256
        self.enc3 = self.encoder.layer2  # 512
        self.enc4 = self.encoder.layer3  # 1024
        self.enc5 = self.encoder.layer4  # 2048
        
        # Load pretrained encoder from self-supervised pretraining
        # Note: Must be done AFTER creating enc1-enc5 since pretrained weights are for these
        if load_pretrained_encoder and os.path.exists(load_pretrained_encoder):
            print(f"🔄 Loading pretrained encoder from: {load_pretrained_encoder}")
            try:
                pretrained_state = torch.load(load_pretrained_encoder, map_location='cpu')
                
                # New format: dictionary with enc1-enc5 keys
                if isinstance(pretrained_state, dict) and 'enc1' in pretrained_state:
                    self.enc1.load_state_dict(pretrained_state['enc1'], strict=False)
                    self.enc2.load_state_dict(pretrained_state['enc2'], strict=False)
                    self.enc3.load_state_dict(pretrained_state['enc3'], strict=False)
                    self.enc4.load_state_dict(pretrained_state['enc4'], strict=False)
                    self.enc5.load_state_dict(pretrained_state['enc5'], strict=False)
                    print(f"✅ Loaded pretrained encoder (all 5 stages)")
                else:
                    # Old format: try to load into encoder directly
                    model_dict = self.encoder.state_dict()
                    pretrained_dict = {k: v for k, v in pretrained_state.items() 
                                     if k in model_dict and v.shape == model_dict[k].shape}
                    model_dict.update(pretrained_dict)
                    self.encoder.load_state_dict(model_dict, strict=False)
                    print(f"✅ Loaded {len(pretrained_dict)} pretrained weights (old format)")
            except Exception as e:
                print(f"⚠️ Could not load pretrained encoder: {e}")

        # Freeze some layers optionally
        if not unfreeze_last_layers:
            for p in self.encoder.parameters():
                p.requires_grad = False
        else:
            for name, p in self.encoder.named_parameters():
                p.requires_grad = False
                if name.startswith("layer4") or name.startswith("layer3"):
                    p.requires_grad = True
        self.enc3 = self.encoder.layer2  # 512
        self.enc4 = self.encoder.layer3  # 1024
        self.enc5 = self.encoder.layer4  # 2048

        # === Decoder (Upsampling with Skip Connections) ===
        self.dec4 = UpBlock(2048 + 1024, 1024)
        self.dec3 = UpBlock(1024 + 512, 512)
        self.dec2 = UpBlock(512 + 256, 256)
        self.dec1 = UpBlock(256 + 64, 128)

        # === Final output layer ===
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, img, depth=None):
        """
        Forward pass for training/inference
        
        Args:
            img: RGB image [B, 3, H, W] in range [-1, 1]
            depth: Depth map [B, 1, H, W] in range [-1, 1] (optional for some tasks)
        
        Returns:
            Restored image [B, 3, H, W] in range [-1, 1]
        """
        # Handle case when depth is not provided (for self-supervised pretraining)
        if depth is None:
            depth = torch.zeros(img.size(0), 1, img.size(2), img.size(3)).to(img.device)
        
        x = torch.cat([img, depth], dim=1)  # [B,4,H,W]

        # === Encoder ===
        e1 = self.enc1(x)   # 1/2
        e2 = self.enc2(e1)  # 1/4
        e3 = self.enc3(e2)  # 1/8
        e4 = self.enc4(e3)  # 1/16
        e5 = self.enc5(e4)  # 1/32

        # === Decoder ===
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # === Output ===
        out = self.final_conv(d1)
        return out

    # === Save & Load Helpers (tetap kompatibel) ===
    def save(self, path, epoch=None, optimizer=None, loss=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'loss': loss
        }
        torch.save(checkpoint, path)
        print(f"💾 Model tersimpan ke {path}")

    @classmethod
    def load(cls, path, device='cpu', **kwargs):
        model = cls(**kwargs)
        model.to(device)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model berhasil dimuat dari {path}")
        else:
            print(f"⚠️ Model {path} tidak ditemukan, menggunakan model baru.")
        return model

    def inference(self, img_tensor, depth_tensor, device='cpu'):
        """Inference mode tanpa gradien"""
        self.eval()
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            depth_tensor = depth_tensor.to(device)
            out = self.forward(img_tensor, depth_tensor)
        return out
