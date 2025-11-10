import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim
from .advanced_damage import RealisticDamageSimulator


def load_image_as_tensor(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BILINEAR)
    return T.ToTensor()(img)  # [0,1]


def load_depth_as_tensor(path, size=None):
    # try load as image PNG/JPG; normalize to [0,1]
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.BILINEAR)
    t = T.ToTensor()(img)  # [0,1]
    return t


def random_mask(h, w, max_patches=3):
    """Generate more conservative random damage mask"""
    mask = np.zeros((h, w), dtype=np.float32)
    n_patches = random.randint(1, max_patches)
    
    for _ in range(n_patches):
        # Smaller damage areas
        rw = random.randint(int(w*0.08), int(w*0.25))
        rh = random.randint(int(h*0.08), int(h*0.25))
        x = random.randint(0, w - rw)
        y = random.randint(0, h - rh)
        mask[y:y+rh, x:x+rw] = 1.0
    
    return torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]


class RestorationDataset(Dataset):
    """
    pairs: list of tuples (images_dir, depth_dir)
    dataset collects all image files inside images_dir and matches depth maps by filename if possible.
    Produces dict with 'damaged', 'ground_truth', 'depth' tensors normalized to [-1,1] (img) and depth in [-1,1].
    
    Features:
    - Realistic damage simulation (cracks, erosion, weathering)
    - Simple damage fallback for compatibility
    """

    def __init__(self, pairs, augment=True, target_size=(512, 512), use_advanced_damage=True):
        self.items = []
        for img_dir, depth_dir in pairs:
            if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(img_dir, fn)
                    # try to find depth file with same name (png or pfm). Prefer png in depth_dir.
                    base = os.path.splitext(fn)[0]
                    cand = None
                    for ext in [".png", ".jpg", ".pfm", ".exr"]:
                        p = os.path.join(depth_dir, base + ext)
                        if os.path.exists(p):
                            cand = p
                            break
                    if cand is None:
                        # fallback: any depth file in folder
                        files = [os.path.join(depth_dir, x) for x in os.listdir(depth_dir) if x.lower().endswith((".png", ".jpg"))]
                        cand = files[0] if files else None
                    if cand:
                        self.items.append((img_path, cand))
        self.augment = augment
        self.target_size = target_size
        self.to_tensor = T.Compose([T.Resize(self.target_size), T.ToTensor()])
        self.use_advanced_damage = use_advanced_damage
        
        # Initialize advanced damage simulator
        if self.use_advanced_damage and self.augment:
            self.damage_simulator = RealisticDamageSimulator()
            print("✅ Using advanced realistic damage simulation")
        else:
            self.damage_simulator = None
            print("ℹ️ Using simple random mask damage")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, depth_path = self.items[idx]
        
        try:
            gt = Image.open(img_path).convert("RGB")
            depth = Image.open(depth_path).convert("L")
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            # Return first item as fallback
            return self.__getitem__(0)

        gt = gt.resize(self.target_size, Image.BILINEAR)
        depth = depth.resize(self.target_size, Image.BILINEAR)

        gt_t = T.ToTensor()(gt)  # [0,1]
        depth_t = T.ToTensor()(depth)  # [0,1]

        # Normalize to [-1,1]
        gt_in = gt_t * 2.0 - 1.0
        depth_in = depth_t * 2.0 - 1.0

        # Create synthetic damage
        if self.augment:
            if self.damage_simulator:
                # Use advanced realistic damage
                damaged_t, mask = self.damage_simulator.apply_realistic_damage(gt_t)
                damaged = damaged_t * 2.0 - 1.0  # Normalize to [-1,1]
            else:
                # Fallback: simple random mask damage
                mask = random_mask(self.target_size[1], self.target_size[0])  # H, W
                noise = torch.randn_like(gt_t) * 0.1 + 0.5
                noise = noise.clamp(0, 1)
                damaged = gt_t * (1 - mask) + noise * mask
                damaged = damaged * 2.0 - 1.0
        else:
            mask = torch.zeros((1, self.target_size[1], self.target_size[0]), dtype=torch.float32)
            damaged = gt_in.clone()
        
        # Sanitize: replace any NaN/Inf values
        for name, tensor in [("damaged", damaged), ("gt", gt_in), ("depth", depth_in)]:
            if not torch.isfinite(tensor).all():
                print(f"⚠️ WARNING: Non-finite values in {name}, replacing with zeros")
                tensor[~torch.isfinite(tensor)] = 0.0
            # Clamp to valid range
            tensor.clamp_(-1.0, 1.0)

        return {"damaged": damaged, "ground_truth": gt_in, "depth": depth_in}


# Inference helper: restore images on disk using trained model
def restore_images_with_model(model, device, images_dir, depth_dir, out_dir, batch_size=4, target_size=(512, 512)):
    """Restore images using trained model and calculate SSIM metrics"""
    # Collect images and matching depth files
    items = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(images_dir, fn)
        base = os.path.splitext(fn)[0]
        cand = None
        for ext in [".png", ".jpg", ".pfm", ".exr"]:
            p = os.path.join(depth_dir, base + ext)
            if os.path.exists(p):
                cand = p
                break
        if cand is None:
            files = [os.path.join(depth_dir, x) for x in os.listdir(depth_dir) if x.lower().endswith((".png", ".jpg"))]
            cand = files[0] if files else None
        if cand:
            items.append((fn, img_path, cand))

    transform_img = T.Compose([T.Resize(target_size), T.ToTensor()])
    transform_depth = T.Compose([T.Resize(target_size), T.ToTensor()])

    metrics_total = {"ssim": 0.0}
    n_images = 0
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            imgs = []
            depths = []
            names = []
            originals = []

            for name, ip, dp in batch:
                im = Image.open(ip).convert("RGB")
                d = Image.open(dp).convert("L")
                
                # Store original for metrics
                originals.append(transform_img(im))
                
                imgs.append(transform_img(im))
                depths.append(transform_depth(d))
                names.append(name)

            imgs_t = torch.stack(imgs, dim=0).to(device)
            depths_t = torch.stack(depths, dim=0).to(device)
            originals_t = torch.stack(originals, dim=0)

            # Normalize to [-1,1]
            imgs_t = imgs_t * 2.0 - 1.0
            depths_t = depths_t * 2.0 - 1.0

            # Forward pass
            out = model(imgs_t, depths_t)
            out = (out.clamp(-1, 1) + 1.0) / 2.0  # Back to [0,1]

            for j in range(out.shape[0]):
                out_img = out[j].cpu()
                
                # Calculate SSIM
                ssim_val = ssim(originals_t[j].unsqueeze(0), 
                              out_img.unsqueeze(0), 
                              data_range=1.0).item()
                metrics_total["ssim"] += ssim_val
                n_images += 1

                # Save restored image
                save_path = os.path.join(out_dir, names[j])
                TF.to_pil_image(out_img).save(save_path)
                
                print(f"✅ Processed {names[j]} - SSIM: {ssim_val:.4f}")

    # Print average metrics
    avg_ssim = metrics_total["ssim"] / max(1, n_images)
    print(f"\n{'='*50}")
    print(f"Restored {len(items)} images -> {out_dir}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"{'='*50}\n")
    
    return {"ssim": avg_ssim}