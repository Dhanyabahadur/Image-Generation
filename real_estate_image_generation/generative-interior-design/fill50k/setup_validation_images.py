#!/usr/bin/env python3
"""
Setup proper validation conditioning images for ControlNet training.

Current issue: image_0.jpg and image_1.jpg are RGB empty room images
Required: Validation images should be conditioning images (segmentation/depth maps)

This script will:
1. Find the conditioning images corresponding to our current validation images
2. Create validation_segmentation_0.png, validation_segmentation_1.png
3. Create validation_depth_0.png, validation_depth_1.png
4. Provide updated training commands
"""

import os
import shutil
import json
from PIL import Image

def find_corresponding_conditioning_images():
    """Find conditioning images that correspond to our current validation images."""
    
    print("🔍 Finding corresponding conditioning images for validation...")
    
    # Load training data to find which images correspond to image_0.jpg and image_1.jpg
    with open('train.jsonl', 'r') as f:
        training_data = [json.loads(line) for line in f]
    
    # Get the first two examples for validation
    validation_examples = training_data[:2]
    
    validation_images = []
    for i, example in enumerate(validation_examples):
        # Extract the base image ID from the target image path
        target_path = example['image']  # e.g., "target_images/12345-67890.jpg"
        base_id = os.path.basename(target_path).replace('.jpg', '')  # "12345-67890"
        
        # Find corresponding conditioning images
        seg_conditioning = f"conditioning_images/{base_id}_segmentation.png"
        depth_conditioning = f"conditioning_images/{base_id}_depth.png"
        
        validation_images.append({
            'id': base_id,
            'target_image': target_path,
            'empty_image': example['input_image'],
            'segmentation_conditioning': seg_conditioning,
            'depth_conditioning': depth_conditioning,
            'caption': example['text']
        })
        
        print(f"📋 Validation {i}:")
        print(f"   ID: {base_id}")
        print(f"   Target: {target_path}")
        print(f"   Empty: {example['input_image']}")
        print(f"   Seg conditioning: {seg_conditioning}")
        print(f"   Depth conditioning: {depth_conditioning}")
    
    return validation_images

def create_validation_conditioning_images(validation_images):
    """Create properly named validation conditioning images."""
    
    print("\n🎨 Creating validation conditioning images...")
    
    for i, img_info in enumerate(validation_images):
        # Create segmentation validation images
        seg_source = img_info['segmentation_conditioning']
        if os.path.exists(seg_source):
            seg_dest = f"validation_segmentation_{i}.png"
            shutil.copy2(seg_source, seg_dest)
            print(f"✅ Created: {seg_dest}")
        else:
            print(f"❌ Missing: {seg_source}")
        
        # Create depth validation images  
        depth_source = img_info['depth_conditioning']
        if os.path.exists(depth_source):
            depth_dest = f"validation_depth_{i}.png"
            shutil.copy2(depth_source, depth_dest)
            print(f"✅ Created: {depth_dest}")
        else:
            print(f"❌ Missing: {depth_source}")

def display_updated_training_commands(validation_images):
    """Display the corrected training commands with proper validation setup."""
    
    print("\n" + "="*80)
    print("🚀 CORRECTED TRAINING COMMANDS")
    print("="*80)
    
    # Extract validation prompts
    validation_prompts = [img['caption'] for img in validation_images]
    
    # Format prompts for command line (escape quotes)
    prompt_0 = validation_prompts[0].replace('"', '\\"')
    prompt_1 = validation_prompts[1].replace('"', '\\"')
    
    print("\n🎨 SEGMENTATION ControlNet Training:")
    print("-" * 50)
    segmentation_cmd = f'''accelerate launch --mixed_precision="bf16" train_controlnet.py \\
  --checkpointing_steps=20000 \\
  --validation_steps=10000 \\
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
  --output_dir=model_segmentation \\
  --resolution=512 \\
  --learning_rate=1e-5 \\
  --validation_image "./validation_segmentation_0.png" "./validation_segmentation_1.png" \\
  --validation_prompt "{prompt_0}" "{prompt_1}" \\
  --train_batch_size=4 \\
  --dataset_name=fill50k.py \\
  --dataset_config_name=segmentation \\
  --controlnet_model_name_or_path "BertChristiaens/controlnet-seg-room" \\
  --report_to wandb \\
  --gradient_accumulation_steps=1 \\
  --mixed_precision="bf16" \\
  --num_train_epochs=10'''
    
    print(segmentation_cmd)
    
    print("\n🔧 DEPTH ControlNet Training:")
    print("-" * 50)
    depth_cmd = f'''accelerate launch --mixed_precision="bf16" train_controlnet.py \\
  --checkpointing_steps=20000 \\
  --validation_steps=10000 \\
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \\
  --output_dir=model_depth \\
  --resolution=512 \\
  --learning_rate=1e-5 \\
  --validation_image "./validation_depth_0.png" "./validation_depth_1.png" \\
  --validation_prompt "{prompt_0}" "{prompt_1}" \\
  --train_batch_size=4 \\
  --dataset_name=fill50k.py \\
  --dataset_config_name=depth \\
  --controlnet_model_name_or_path "lllyasviel/sd-controlnet-depth" \\
  --report_to wandb \\
  --gradient_accumulation_steps=1 \\
  --mixed_precision="bf16" \\
  --num_train_epochs=10'''
    
    print(depth_cmd)
    
    print("\n" + "="*80)
    print("💡 KEY CHANGES:")
    print("✅ Validation images are now CONDITIONING maps (seg/depth)")
    print("✅ Validation prompts match the actual room captions")
    print("✅ Separate configs for segmentation vs depth training")
    print("✅ Proper dataset_config_name parameter usage")
    print("="*80)

def main():
    """Main function to setup validation images."""
    
    print("🔧 Setting up proper validation conditioning images for ControlNet training")
    print("="*80)
    
    # Find corresponding conditioning images
    validation_images = find_corresponding_conditioning_images()
    
    # Create validation conditioning images
    create_validation_conditioning_images(validation_images)
    
    # Show updated training commands
    display_updated_training_commands(validation_images)
    
    print(f"\n✅ Validation setup complete!")
    print(f"📁 Created validation conditioning images in: {os.getcwd()}")
    print(f"🚀 Ready to train with proper ControlNet validation!")

if __name__ == "__main__":
    main() 
