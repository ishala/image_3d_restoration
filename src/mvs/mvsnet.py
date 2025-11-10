import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ...existing code...
class MVSNet(nn.Module):
    def __init__(self, feat_dim=32):
        super().__init__()
        # very small example encoder / decoder (ganti sesuai MVSNet nyata)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, 1, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        return self.head(f)

def save_checkpoint(model: nn.Module, optimizer, epoch:int, path:str, meta:dict=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "meta": meta or {}
    }
    torch.save(payload, path)

def load_checkpoint(model: nn.Module, optimizer, path:str, map_location=None):
    ck = torch.load(path, map_location=map_location)
    model.load_state_dict(ck["state_dict"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return ck.get("epoch", 0), ck.get("meta", {})

def train_mvsnet(model: nn.Module, dataset, epochs, device, ckpt_dir):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    history = {"loss": []}
    for e in range(1, epochs+1):
        model.train()
        running = 0.0
        for batch in loader:
            imgs = batch["image"].to(device)         # sesuaikan key
            gt = batch.get("depth", None)
            if gt is None:
                continue
            gt = gt.to(device)
            opt.zero_grad()
            pred = model(imgs)
            loss = nn.functional.mse_loss(pred, gt)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running / len(loader)
        history["loss"].append(avg)
        # save each epoch
        save_checkpoint(model, opt, e, os.path.join(ckpt_dir, f"mvsnet_epoch{e}.pth"),
                        meta={"history": history})
        print(f"Epoch {e} avg_loss={avg:.4f}")
    return history
# ...existing code...