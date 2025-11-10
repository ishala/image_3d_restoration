# datasets/augmentations.py
import os
import random
from PIL import Image, ImageEnhance

def photometric_jitter(image: Image.Image) -> Image.Image:
    brightness = random.uniform(0.7, 1.3)
    contrast = random.uniform(0.8, 1.2)
    color = random.uniform(0.9, 1.1)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(color)
    return image

def pose_perturbation(image: Image.Image, max_angle: float = 20):
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BILINEAR, expand=False)

def augment_image(image: Image.Image):
    """Kombinasikan photometric jitter + rotasi kecil"""
    img = photometric_jitter(image)
    img = pose_perturbation(img)
    return img

def augment_dataset(images_dir: str, num_aug: int = 3):
    """
    Lakukan augmentasi pada semua gambar dalam folder.
    Hasil augmentasi disimpan di folder yang sama dengan prefix 'aug_'.
    
    Args:
        images_dir: Path ke direktori yang berisi gambar untuk diaugmentasi
        num_aug: Jumlah augmentasi per gambar (default: 3)
    """
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith((".jpg", ".png"))]
    
    total_augmented = 0
    for f in image_files:
        img_path = os.path.join(images_dir, f)
        try:
            img = Image.open(img_path).convert("RGB")
            for i in range(num_aug):
                aug_img = augment_image(img)
                new_name = f"aug_{os.path.splitext(f)[0]}_{i}.jpg"
                aug_path = os.path.join(images_dir, new_name)
                aug_img.save(aug_path, quality=95)  # Simpan dengan kualitas tinggi
                total_augmented += 1
        except Exception as e:
            print(f"⚠️ Gagal mengaugmentasi {f}: {str(e)}")
            continue
    
    print(f"🧠 Augmentasi selesai → {total_augmented} gambar sintetis ditambahkan dari {len(image_files)} gambar asli.")
