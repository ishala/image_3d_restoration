"""
Self-Supervised Pretraining for Restoration Model
Helps model learn useful representations from unlabeled data before fine-tuning
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np


class RotationPredictionTask:
    """
    Self-supervised task: Predict rotation angle
    Helps encoder learn spatial features and orientation
    """
    
    def __init__(self, model, device, checkpoints_dir="checkpoints"):
        self.model = model
        self.device = device
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Add rotation prediction head
        # Get encoder output channels (should be from model.encoder)
        # Assuming ResNet50 encoder outputs 2048 channels
        self.rotation_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4)  # 4 classes: 0°, 90°, 180°, 270°
        ).to(device)
        
        # Optimizer for rotation task
        # Use all encoder stages (enc1-enc5)
        encoder_params = (
            list(self.model.enc1.parameters()) +
            list(self.model.enc2.parameters()) +
            list(self.model.enc3.parameters()) +
            list(self.model.enc4.parameters()) +
            list(self.model.enc5.parameters())
        )
        self.optimizer = optim.Adam(
            encoder_params + list(self.rotation_head.parameters()),
            lr=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        print("✅ Rotation prediction task initialized")
    
    def create_rotated_batch(self, images):
        """
        Create batch with random rotations
        
        Args:
            images: Tensor [B, C, H, W]
        
        Returns:
            rotated_images: Tensor [B, C, H, W]
            rotation_labels: Tensor [B] with values 0-3
        """
        batch_size = images.size(0)
        angles = [0, 90, 180, 270]
        
        rotated_images = []
        rotation_labels = []
        
        for img in images:
            # Random rotation
            angle_idx = torch.randint(0, 4, (1,)).item()
            angle = angles[angle_idx]
            
            # Rotate image
            rotated = TF.rotate(img, angle)
            
            rotated_images.append(rotated)
            rotation_labels.append(angle_idx)
        
        rotated_images = torch.stack(rotated_images)
        rotation_labels = torch.tensor(rotation_labels, dtype=torch.long)
        
        return rotated_images, rotation_labels
    
    def train_epoch(self, unlabeled_loader, epoch):
        """Train one epoch on rotation prediction task"""
        self.model.train()
        self.rotation_head.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(unlabeled_loader):
            # Get images (can be from 'image', 'damaged', or any key)
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('damaged', batch.get('ground_truth')))
            else:
                images = batch
            
            images = images.to(self.device)
            
            # Create rotated versions
            rotated_images, rotation_labels = self.create_rotated_batch(images)
            rotated_images = rotated_images.to(self.device)
            rotation_labels = rotation_labels.to(self.device)
            
            # Add dummy depth channel (encoder expects 4 channels: RGB + Depth)
            # During pretraining we only have RGB, so add zeros for depth
            batch_size, _, h, w = rotated_images.shape
            dummy_depth = torch.zeros(batch_size, 1, h, w, device=self.device)
            rotated_images_4ch = torch.cat([rotated_images, dummy_depth], dim=1)
            
            # Forward pass through encoder stages to get features
            # Use the encoder stages from restoration model
            e1 = self.model.enc1(rotated_images_4ch)   # [B, 64, H/2, W/2]
            e2 = self.model.enc2(e1)                    # [B, 256, H/4, W/4]
            e3 = self.model.enc3(e2)                    # [B, 512, H/8, W/8]
            e4 = self.model.enc4(e3)                    # [B, 1024, H/16, W/16]
            features = self.model.enc5(e4)              # [B, 2048, H/32, W/32]
            
            # Predict rotation
            rotation_logits = self.rotation_head(features)
            
            # Compute loss
            loss = self.criterion(rotation_logits, rotation_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients for all encoder stages
            encoder_params = (
                list(self.model.enc1.parameters()) +
                list(self.model.enc2.parameters()) +
                list(self.model.enc3.parameters()) +
                list(self.model.enc4.parameters()) +
                list(self.model.enc5.parameters())
            )
            torch.nn.utils.clip_grad_norm_(
                encoder_params + list(self.rotation_head.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = rotation_logits.max(1)
            correct += predicted.eq(rotation_labels).sum().item()
            total += rotation_labels.size(0)
            
            # Print progress
            if batch_idx % 10 == 0:
                acc = 100.0 * correct / total if total > 0 else 0
                print(f"  Batch [{batch_idx}/{len(unlabeled_loader)}] "
                      f"Loss: {loss.item():.4f} Acc: {acc:.2f}%")
        
        avg_loss = total_loss / len(unlabeled_loader)
        avg_acc = 100.0 * correct / total if total > 0 else 0
        
        print(f"Epoch {epoch} - Rotation Prediction - Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}%")
        
        return avg_loss, avg_acc
    
    def pretrain(self, unlabeled_loader, epochs=10):
        """
        Pretrain encoder using rotation prediction task
        
        Args:
            unlabeled_loader: DataLoader with unlabeled images
            epochs: Number of pretraining epochs
        """
        print("\n" + "="*60)
        print("SELF-SUPERVISED PRETRAINING: Rotation Prediction")
        print("="*60)
        
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            loss, acc = self.train_epoch(unlabeled_loader, epoch)
            
            # Save best encoder
            if acc > best_acc:
                best_acc = acc
                # Save all encoder stages
                encoder_state = {
                    'enc1': self.model.enc1.state_dict(),
                    'enc2': self.model.enc2.state_dict(),
                    'enc3': self.model.enc3.state_dict(),
                    'enc4': self.model.enc4.state_dict(),
                    'enc5': self.model.enc5.state_dict(),
                }
                torch.save(
                    encoder_state,
                    os.path.join(self.checkpoints_dir, "pretrained_encoder.pth")
                )
                print(f"✅ Best encoder saved (acc: {acc:.2f}%)")
        
        print("\n" + "="*60)
        print(f"✅ Pretraining completed! Best accuracy: {best_acc:.2f}%")
        print("="*60)
        
        # Save final rotation head (optional, for analysis)
        torch.save(
            self.rotation_head.state_dict(),
            os.path.join(self.checkpoints_dir, "rotation_head.pth")
        )


class JigsawPuzzleTask:
    """
    Self-supervised task: Solve jigsaw puzzles
    Helps encoder learn part-whole relationships and spatial reasoning
    """
    
    def __init__(self, model, device, grid_size=3, checkpoints_dir="checkpoints"):
        self.model = model
        self.device = device
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Jigsaw puzzle prediction head
        # Predicts permutation index (there are 9! = 362880 possible for 3x3, we'll use subset)
        # Simplified: use 100 most common permutations
        self.num_permutations = 100
        
        self.jigsaw_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, self.num_permutations)
        ).to(device)
        
        # Generate permutation set
        self.permutations = self._generate_permutations(self.num_permutations)
        
        # Optimizer for jigsaw task
        # Use all encoder stages (enc1-enc5)
        encoder_params = (
            list(self.model.enc1.parameters()) +
            list(self.model.enc2.parameters()) +
            list(self.model.enc3.parameters()) +
            list(self.model.enc4.parameters()) +
            list(self.model.enc5.parameters())
        )
        self.optimizer = optim.Adam(
            encoder_params + list(self.jigsaw_head.parameters()),
            lr=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"✅ Jigsaw puzzle task initialized ({grid_size}x{grid_size} grid)")
    
    def _generate_permutations(self, num_perms):
        """Generate random permutations for jigsaw puzzles"""
        permutations = []
        for _ in range(num_perms):
            perm = torch.randperm(self.num_patches)
            permutations.append(perm)
        return permutations
    
    def create_jigsaw(self, image, perm_idx):
        """
        Create jigsaw puzzle from image
        
        Args:
            image: Tensor [C, H, W]
            perm_idx: Index of permutation to use
        
        Returns:
            puzzled_image: Tensor [C, H, W] with shuffled patches
        """
        C, H, W = image.shape
        patch_h = H // self.grid_size
        patch_w = W // self.grid_size
        
        # Extract patches
        patches = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y1 = i * patch_h
                y2 = (i + 1) * patch_h
                x1 = j * patch_w
                x2 = (j + 1) * patch_w
                patch = image[:, y1:y2, x1:x2]
                patches.append(patch)
        
        # Shuffle patches according to permutation
        perm = self.permutations[perm_idx]
        shuffled_patches = [patches[i] for i in perm]
        
        # Reconstruct image
        rows = []
        for i in range(self.grid_size):
            row_patches = shuffled_patches[i * self.grid_size:(i + 1) * self.grid_size]
            row = torch.cat(row_patches, dim=2)  # Concat along width
            rows.append(row)
        
        puzzled_image = torch.cat(rows, dim=1)  # Concat along height
        
        return puzzled_image
    
    def pretrain(self, unlabeled_loader, epochs=10):
        """
        Pretrain encoder using jigsaw puzzle task
        
        Args:
            unlabeled_loader: DataLoader with unlabeled images
            epochs: Number of pretraining epochs
        """
        print("\n" + "="*60)
        print("SELF-SUPERVISED PRETRAINING: Jigsaw Puzzle")
        print("="*60)
        
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 40)
            
            self.model.train()
            self.jigsaw_head.train()
            
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, batch in enumerate(unlabeled_loader):
                # Get images
                if isinstance(batch, dict):
                    images = batch.get('image', batch.get('damaged', batch.get('ground_truth')))
                else:
                    images = batch
                
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # Create jigsaw puzzles
                puzzled_images = []
                perm_labels = []
                
                for img in images:
                    perm_idx = torch.randint(0, self.num_permutations, (1,)).item()
                    puzzled = self.create_jigsaw(img, perm_idx)
                    puzzled_images.append(puzzled)
                    perm_labels.append(perm_idx)
                
                puzzled_images = torch.stack(puzzled_images).to(self.device)
                perm_labels = torch.tensor(perm_labels, dtype=torch.long).to(self.device)
                
                # Add dummy depth channel (encoder expects 4 channels: RGB + Depth)
                batch_size_jigsaw, _, h, w = puzzled_images.shape
                dummy_depth = torch.zeros(batch_size_jigsaw, 1, h, w, device=self.device)
                puzzled_images_4ch = torch.cat([puzzled_images, dummy_depth], dim=1)
                
                # Forward pass through encoder stages to get features
                e1 = self.model.enc1(puzzled_images_4ch)   # [B, 64, H/2, W/2]
                e2 = self.model.enc2(e1)                    # [B, 256, H/4, W/4]
                e3 = self.model.enc3(e2)                    # [B, 512, H/8, W/8]
                e4 = self.model.enc4(e3)                    # [B, 1024, H/16, W/16]
                features = self.model.enc5(e4)              # [B, 2048, H/32, W/32]
                
                jigsaw_logits = self.jigsaw_head(features)
                
                # Compute loss
                loss = self.criterion(jigsaw_logits, perm_labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients for all encoder stages
                encoder_params = (
                    list(self.model.enc1.parameters()) +
                    list(self.model.enc2.parameters()) +
                    list(self.model.enc3.parameters()) +
                    list(self.model.enc4.parameters()) +
                    list(self.model.enc5.parameters())
                )
                torch.nn.utils.clip_grad_norm_(
                    encoder_params + list(self.jigsaw_head.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = jigsaw_logits.max(1)
                correct += predicted.eq(perm_labels).sum().item()
                total += perm_labels.size(0)
                
                if batch_idx % 10 == 0:
                    acc = 100.0 * correct / total if total > 0 else 0
                    print(f"  Batch [{batch_idx}/{len(unlabeled_loader)}] "
                          f"Loss: {loss.item():.4f} Acc: {acc:.2f}%")
            
            avg_loss = total_loss / len(unlabeled_loader)
            avg_acc = 100.0 * correct / total if total > 0 else 0
            
            print(f"Epoch {epoch} - Jigsaw Puzzle - Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}%")
            
            # Save best encoder
            if avg_acc > best_acc:
                best_acc = avg_acc
                # Save all encoder stages
                encoder_state = {
                    'enc1': self.model.enc1.state_dict(),
                    'enc2': self.model.enc2.state_dict(),
                    'enc3': self.model.enc3.state_dict(),
                    'enc4': self.model.enc4.state_dict(),
                    'enc5': self.model.enc5.state_dict(),
                }
                torch.save(
                    encoder_state,
                    os.path.join(self.checkpoints_dir, "pretrained_encoder_jigsaw.pth")
                )
                print(f"✅ Best encoder saved (acc: {avg_acc:.2f}%)")
        
        print("\n" + "="*60)
        print(f"✅ Pretraining completed! Best accuracy: {best_acc:.2f}%")
        print("="*60)


class UnlabeledImageDataset(Dataset):
    """Dataset for unlabeled images (for self-supervised pretraining)"""
    
    def __init__(self, image_dir, transform=None, target_size=(512, 512)):
        self.image_dir = image_dir
        self.target_size = target_size
        
        # Find all images
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✅ Found {len(self.image_paths)} unlabeled images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {'image': image_tensor}


def run_self_supervised_pretraining(model, device, unlabeled_data_dir, 
                                    task='rotation', epochs=10, batch_size=16,
                                    checkpoints_dir="checkpoints"):
    """
    Run self-supervised pretraining
    
    Args:
        model: Restoration model to pretrain
        device: torch device
        unlabeled_data_dir: Directory with unlabeled images
        task: 'rotation' or 'jigsaw'
        epochs: Number of epochs
        batch_size: Batch size
        checkpoints_dir: Directory to save checkpoints
    """
    # Create unlabeled dataset
    dataset = UnlabeledImageDataset(unlabeled_data_dir)
    
    if len(dataset) == 0:
        print("❌ No unlabeled images found!")
        return
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Select task
    if task == 'rotation':
        trainer = RotationPredictionTask(model, device, checkpoints_dir)
        trainer.pretrain(loader, epochs)
    elif task == 'jigsaw':
        trainer = JigsawPuzzleTask(model, device, grid_size=3, checkpoints_dir=checkpoints_dir)
        trainer.pretrain(loader, epochs)
    else:
        raise ValueError(f"Unknown task: {task}. Use 'rotation' or 'jigsaw'")
