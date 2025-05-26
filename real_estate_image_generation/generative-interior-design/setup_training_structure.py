import os
import shutil
import json
from pathlib import Path

def setup_correct_training_structure():
    """Set up the CORRECT folder structure for training.
    
    Training goal: Empty room + prompt -> Furnished room
    So we need:
    - Input images: Empty/cleaned room images
    - Conditioning: Segmentation/depth of empty rooms  
    - Target images: Original furnished room images
    - Text: Captions describing the furnished rooms
    """
    
    # Create main training directory
    training_dir = "fill50k"
    os.makedirs(training_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(training_dir, "images")  # Will contain EMPTY room images
    conditioning_images_dir = os.path.join(training_dir, "conditioning_images")  # Seg/depth of empty rooms
    target_images_dir = os.path.join(training_dir, "target_images")  # Original furnished images
    diffusers_dir = os.path.join(training_dir, "diffusers")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(conditioning_images_dir, exist_ok=True)
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(diffusers_dir, exist_ok=True)
    
    print(f"✅ Created CORRECTED directory structure:")
    print(f"   📁 {training_dir}/")
    print(f"   ├── 📁 images/ (EMPTY room images)")
    print(f"   ├── 📁 conditioning_images/ (segmentation/depth of empty rooms)")
    print(f"   ├── 📁 target_images/ (original FURNISHED room images)")
    print(f"   └── 📁 diffusers/")
    
    # Load captions to get list of successfully processed images
    captions_file = "room_captions_robust.json"
    if not os.path.exists(captions_file):
        print(f"❌ Error: {captions_file} not found!")
        return
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    print(f"📋 Found {len(captions_data)} captioned images")
    
    # Source directories
    images_base_dir = "dataset_gathering/data/images_price/"  # Original furnished images
    clean_images_dir = "get_empty_room/clean_images/"  # Empty images and conditioning
    
    copied_empty = 0
    copied_furnished = 0
    copied_segmentation = 0
    copied_depth = 0
    
    for image_id, caption_info in captions_data.items():
        # 1. Copy EMPTY room images to images/ folder (these are the inputs)
        empty_source = os.path.join(clean_images_dir, f"{image_id}_clean.png")
        if os.path.exists(empty_source):
            empty_dest = os.path.join(images_dir, f"{image_id}.jpg")  # Keep .jpg for consistency
            
            # Convert PNG to JPG for consistency
            from PIL import Image
            img = Image.open(empty_source).convert('RGB')
            img.save(empty_dest, 'JPEG', quality=95)
            copied_empty += 1
        
        # 2. Copy segmentation and depth of EMPTY rooms to conditioning_images/
        seg_source = os.path.join(clean_images_dir, f"{image_id}_segmentation.png")
        if os.path.exists(seg_source):
            seg_dest = os.path.join(conditioning_images_dir, f"{image_id}_segmentation.png")
            shutil.copy2(seg_source, seg_dest)
            copied_segmentation += 1
        
        depth_source = os.path.join(clean_images_dir, f"{image_id}_depth.png")
        if os.path.exists(depth_source):
            depth_dest = os.path.join(conditioning_images_dir, f"{image_id}_depth.png")
            shutil.copy2(depth_source, depth_dest)
            copied_depth += 1
        
        # 3. Copy ORIGINAL FURNISHED images to target_images/ (these are the targets)
        original_image_path = None
        for subdir in os.listdir(images_base_dir):
            subdir_path = os.path.join(images_base_dir, subdir)
            if os.path.isdir(subdir_path):
                potential_path = os.path.join(subdir_path, f"{image_id}.jpg")
                if os.path.exists(potential_path):
                    original_image_path = potential_path
                    break
        
        if original_image_path:
            furnished_dest = os.path.join(target_images_dir, f"{image_id}.jpg")
            shutil.copy2(original_image_path, furnished_dest)
            copied_furnished += 1
    
    print(f"\n📊 Copy Summary:")
    print(f"🏠 Empty room images: {copied_empty}")
    print(f"🖼️  Furnished room images: {copied_furnished}")
    print(f"🎨 Segmentation maps: {copied_segmentation}")
    print(f"📐 Depth maps: {copied_depth}")
    
    # Create sample validation images (empty rooms for validation)
    empty_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    if len(empty_files) >= 2:
        # Copy first two EMPTY images as validation samples
        shutil.copy2(os.path.join(images_dir, empty_files[0]), 
                     os.path.join(training_dir, "image_0.jpg"))
        shutil.copy2(os.path.join(images_dir, empty_files[1]), 
                     os.path.join(training_dir, "image_1.jpg"))
        print(f"🖼️  Created validation images: image_0.jpg, image_1.jpg (empty rooms)")
    
    print(f"\n🎯 CORRECTED Training structure ready!")
    print(f"📝 Training goal: Empty room + prompt → Furnished room")
    print(f"📁 Location: {training_dir}/")
    
    return training_dir

if __name__ == "__main__":
    setup_correct_training_structure()