"""
Advanced Damage Simulation for Artifact Restoration
Simulates realistic damage patterns: cracks, erosion, weathering, color degradation
"""
import cv2
import numpy as np
import torch
import random
from PIL import Image


class RealisticDamageSimulator:
    """
    Simulate realistic damage patterns found in archaeological artifacts:
    - Cracks and fractures
    - Missing pieces (erosion)
    - Color degradation (weathering)
    - Surface noise (dirt, dust)
    - Edge erosion
    """
    
    def __init__(self, damage_probability=0.8):
        """
        Args:
            damage_probability: Probability of applying each damage type
        """
        self.damage_prob = damage_probability
    
    def create_crack_pattern(self, h, w, num_cracks=None):
        """
        Generate realistic crack patterns
        
        Args:
            h, w: Image dimensions
            num_cracks: Number of cracks (random if None)
        
        Returns:
            Crack mask [H, W] with values 0-1
        """
        mask = np.zeros((h, w), dtype=np.float32)
        
        if num_cracks is None:
            num_cracks = random.randint(1, 4)
        
        for _ in range(num_cracks):
            # Random start and end points
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)
            
            # Draw crack with random thickness
            thickness = random.randint(2, 8)
            cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness)
            
            # Add branching cracks (30% chance)
            if random.random() < 0.3:
                # Branch from midpoint
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                branch_x = mid_x + random.randint(-w//4, w//4)
                branch_y = mid_y + random.randint(-h//4, h//4)
                branch_thickness = max(1, thickness - 2)
                cv2.line(mask, (mid_x, mid_y), (branch_x, branch_y), 1.0, branch_thickness)
        
        # Smooth crack edges
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        return mask
    
    def create_missing_pieces(self, h, w, num_pieces=None):
        """
        Generate missing piece patterns (irregular shapes)
        
        Args:
            h, w: Image dimensions
            num_pieces: Number of missing pieces (random if None)
        
        Returns:
            Mask [H, W] with values 0-1
        """
        mask = np.zeros((h, w), dtype=np.float32)
        
        if num_pieces is None:
            num_pieces = random.randint(1, 3)
        
        for _ in range(num_pieces):
            # Random position and size
            x_center = random.randint(50, w - 50)
            y_center = random.randint(50, h - 50)
            
            # Create irregular polygon
            num_vertices = random.randint(5, 8)
            radius_base = random.randint(30, 80)
            
            points = []
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                radius = radius_base + random.randint(-20, 20)
                x = int(x_center + radius * np.cos(angle))
                y = int(y_center + radius * np.sin(angle))
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 1.0)
        
        # Smooth edges for realistic erosion
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def create_edge_erosion(self, h, w, edge=None):
        """
        Generate edge erosion pattern
        
        Args:
            h, w: Image dimensions
            edge: Which edge to erode ('top', 'bottom', 'left', 'right', or None for random)
        
        Returns:
            Mask [H, W] with values 0-1
        """
        mask = np.zeros((h, w), dtype=np.float32)
        
        if edge is None:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
        
        # Random erosion depth
        erosion_depth = random.randint(30, 120)
        
        # Create irregular edge
        if edge == 'top':
            for x in range(w):
                depth = erosion_depth + random.randint(-20, 20)
                mask[:depth, x] = 1.0
        elif edge == 'bottom':
            for x in range(w):
                depth = erosion_depth + random.randint(-20, 20)
                mask[-depth:, x] = 1.0
        elif edge == 'left':
            for y in range(h):
                depth = erosion_depth + random.randint(-20, 20)
                mask[y, :depth] = 1.0
        else:  # right
            for y in range(h):
                depth = erosion_depth + random.randint(-20, 20)
                mask[y, -depth:] = 1.0
        
        # Smooth transition
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def apply_color_degradation(self, image_tensor):
        """
        Apply color degradation (aging, weathering)
        
        Args:
            image_tensor: Tensor [C, H, W] in range [0, 1]
        
        Returns:
            Degraded image tensor [C, H, W]
        """
        # Convert to numpy
        img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        degraded = img_np.copy()
        
        # Random degradation type
        degradation_type = random.choice(['sepia', 'gray', 'fade', 'yellowing'])
        
        if degradation_type == 'sepia':
            # Sepia tone (aging)
            kernel = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
            degraded = cv2.transform(img_np, kernel)
            degraded = np.clip(degraded, 0, 255)
        
        elif degradation_type == 'gray':
            # Partial desaturation
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            alpha = random.uniform(0.3, 0.7)
            degraded = (alpha * gray + (1 - alpha) * img_np).astype(np.uint8)
        
        elif degradation_type == 'fade':
            # Fading (brightness reduction)
            factor = random.uniform(0.6, 0.9)
            degraded = (img_np * factor).astype(np.uint8)
        
        else:  # yellowing
            # Add yellow tint
            yellow_tint = np.array([20, 20, 0], dtype=np.uint8)
            degraded = np.clip(img_np.astype(np.int16) + yellow_tint, 0, 255).astype(np.uint8)
        
        # Convert back to tensor
        degraded_tensor = torch.from_numpy(degraded).permute(2, 0, 1).float() / 255.0
        
        return degraded_tensor
    
    def apply_surface_noise(self, image_tensor):
        """
        Apply surface noise (dirt, dust, grain)
        
        Args:
            image_tensor: Tensor [C, H, W] in range [0, 1]
        
        Returns:
            Noisy image tensor [C, H, W]
        """
        img = image_tensor.clone()
        
        # Random noise type
        noise_type = random.choice(['gaussian', 'salt_pepper', 'grain'])
        
        if noise_type == 'gaussian':
            # Gaussian noise (dust)
            noise = torch.randn_like(img) * random.uniform(0.02, 0.08)
            img = img + noise
        
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise (spots)
            prob = random.uniform(0.01, 0.05)
            mask = torch.rand_like(img[0:1])
            salt = (mask < prob / 2).float()
            pepper = (mask > 1 - prob / 2).float()
            img = img * (1 - salt - pepper) + salt
        
        else:  # grain
            # Film grain
            grain = torch.randn_like(img) * random.uniform(0.01, 0.03)
            img = img + grain
        
        return torch.clamp(img, 0, 1)
    
    def generate_combined_damage_mask(self, h, w):
        """
        Generate combined damage mask with multiple realistic patterns
        
        Args:
            h, w: Image dimensions
        
        Returns:
            Combined mask [1, H, W] with values 0-1
        """
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Apply cracks (40% chance)
        if random.random() < 0.4:
            crack_mask = self.create_crack_pattern(h, w)
            mask = np.maximum(mask, crack_mask * 0.8)  # Cracks are partial damage
        
        # Apply missing pieces (30% chance)
        if random.random() < 0.3:
            missing_mask = self.create_missing_pieces(h, w)
            mask = np.maximum(mask, missing_mask)
        
        # Apply edge erosion (25% chance)
        if random.random() < 0.25:
            erosion_mask = self.create_edge_erosion(h, w)
            mask = np.maximum(mask, erosion_mask)
        
        # Ensure some damage if nothing applied
        if mask.max() < 0.1:
            # Fallback to simple random patches
            num_patches = random.randint(1, 2)
            for _ in range(num_patches):
                x = random.randint(0, w - 100)
                y = random.randint(0, h - 100)
                w_patch = random.randint(50, 150)
                h_patch = random.randint(50, 150)
                mask[y:y+h_patch, x:x+w_patch] = 1.0
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return torch.from_numpy(mask).unsqueeze(0)
    
    def apply_realistic_damage(self, image_tensor):
        """
        Apply full realistic damage pipeline
        
        Args:
            image_tensor: Tensor [C, H, W] in range [0, 1]
        
        Returns:
            damaged: Damaged image tensor [C, H, W] in range [0, 1]
            mask: Damage mask [1, H, W] with values 0-1
        """
        C, H, W = image_tensor.shape
        
        # Start with original image
        damaged = image_tensor.clone()
        
        # 1. Color degradation (70% chance)
        if random.random() < 0.7:
            damaged = self.apply_color_degradation(damaged)
        
        # 2. Surface noise (60% chance)
        if random.random() < 0.6:
            damaged = self.apply_surface_noise(damaged)
        
        # 3. Generate structural damage mask
        mask = self.generate_combined_damage_mask(H, W)
        
        # 4. Apply mask-based damage
        # Create noise/texture for damaged areas
        noise = torch.rand_like(damaged) * 0.5  # Dark noise
        
        # Blend damaged areas
        mask_3ch = mask.repeat(3, 1, 1)
        damaged = damaged * (1 - mask_3ch) + noise * mask_3ch
        
        return damaged, mask


def apply_advanced_damage(gt_tensor, simulator=None):
    """
    Convenience function to apply advanced damage
    
    Args:
        gt_tensor: Ground truth image [C, H, W] in range [0, 1]
        simulator: RealisticDamageSimulator instance (creates new if None)
    
    Returns:
        damaged: Damaged image [C, H, W]
        mask: Damage mask [1, H, W]
    """
    if simulator is None:
        simulator = RealisticDamageSimulator()
    
    return simulator.apply_realistic_damage(gt_tensor)
