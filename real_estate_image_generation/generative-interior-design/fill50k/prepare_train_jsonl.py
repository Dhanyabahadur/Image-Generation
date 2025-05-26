# import os
# import json

# def clean_caption(caption):
#     """Clean caption by removing trailing punctuation and excess characters."""
#     caption = caption.strip().strip('"')
#     if caption.endswith(","):
#         caption = caption[:-1]
#     last_period_index = caption.rfind('.')
#     last_comma_index = caption.rfind(',')
#     last_index = max(last_period_index, last_comma_index)
#     if last_index != -1:
#         caption = caption[:last_index]
#     return caption

# def process_caption_line(line, images_folder, conditioning_images_folder, parent_folder):
#     """Process a line from the captions file."""
#     if "{" in line or "}" in line:
#         return None
#     filename, caption = line.split(':', 1)
#     filename, caption = filename.strip().strip('"'), clean_caption(caption)
#     image_path = os.path.join(images_folder, filename)
#     conditioning_image_path = os.path.join(conditioning_images_folder, filename.replace(".jpg", "segmentation.png"))
#     if os.path.exists(image_path) and os.path.exists(conditioning_image_path):
#         image_path = image_path.replace(parent_folder, '').lstrip('/')
#         conditioning_image_path = conditioning_image_path.replace(parent_folder, '').lstrip('/')
#         return {"text": caption, "image": image_path, "conditioning_image": conditioning_image_path}
#     return None

# def process_captions_file(captions_file, images_folder, conditioning_images_folder, output_file):
#     """Process the captions file and write JSON objects to the output file."""
#     parent_folder = os.path.dirname(images_folder)
#     with open(output_file, 'w') as writer, open(captions_file, 'r') as f:
#         for line in f:
#             json_object = process_caption_line(line, images_folder, conditioning_images_folder, parent_folder)
#             if json_object:
#                 writer.write(json.dumps(json_object) + '\n')

# def main():
#     # Define paths
#     images_folder = "/mnt/data1/nick/depth/fill50k/images"
#     conditioning_images_folder = "/mnt/data1/nick/depth/fill50k/conditioning_images"
#     captions_file = "/mnt/data1/nick/depth/fill50k/bnb-dataset/get_captions/captions.json"
#     output_file = "/mnt/data1/nick/depth/fill50k/train.jsonl"
#     # Process captions file
#     process_captions_file(captions_file, images_folder, conditioning_images_folder, output_file)

# if __name__ == "__main__":
#     main()


import os
import json

def clean_caption(caption):
    """Clean caption by removing trailing punctuation and excess characters."""
    caption = caption.strip().strip('"')
    if caption.endswith(","):
        caption = caption[:-1]
    last_period_index = caption.rfind('.')
    last_comma_index = caption.rfind(',')
    last_index = max(last_period_index, last_comma_index)
    if last_index != -1:
        caption = caption[:last_index]
    return caption

def process_captions_json_corrected(captions_file, images_folder, conditioning_images_folder, target_images_folder, output_file, conditioning_type="segmentation"):
    """Process the captions JSON file for CORRECT training setup.
    
    Training goal: Empty room + prompt -> Furnished room
    
    Structure:
    - image: Empty room (input)
    - conditioning_image: Segmentation/depth of empty room (guidance)
    - target_image: Furnished room (what we want to generate)
    - text: Caption describing the furnished room (what we want to achieve)
    """
    
    print(f"Loading captions from: {captions_file}")
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    print(f"Found {len(captions_data)} captioned images")
    print(f"Training setup: Empty room + {conditioning_type} + prompt → Furnished room")
    
    processed_count = 0
    skipped_count = 0
    
    with open(output_file, 'w') as writer:
        for image_id, caption_info in captions_data.items():
            # Get caption (describes the desired furnished room)
            if isinstance(caption_info, dict):
                caption = caption_info.get('caption', '')
            else:
                caption = str(caption_info)
            
            caption = clean_caption(caption)
            
            # Construct file paths
            image_filename = f"{image_id}.jpg"
            
            # Input: Empty room image
            input_image_path = os.path.join(images_folder, image_filename)
            
            # Conditioning: Segmentation/depth of empty room
            if conditioning_type == "segmentation":
                conditioning_filename = f"{image_id}_segmentation.png"
            else:  # depth
                conditioning_filename = f"{image_id}_depth.png"
            conditioning_image_path = os.path.join(conditioning_images_folder, conditioning_filename)
            
            # Target: Original furnished room
            target_image_path = os.path.join(target_images_folder, image_filename)
            
            # Check if all files exist
            if (os.path.exists(input_image_path) and 
                os.path.exists(conditioning_image_path) and 
                os.path.exists(target_image_path)):
                
                # Create relative paths for the JSON
                relative_input_path = f"images/{image_filename}"
                relative_conditioning_path = f"conditioning_images/{conditioning_filename}"
                relative_target_path = f"target_images/{image_filename}"
                
                # NOTE: For ControlNet training, we typically use 'image' as the target
                # The conditioning comes through the conditioning_image
                # So we structure it as: input=empty room, target=furnished room
                json_object = {
                    "text": caption,  # Description of desired furnished room
                    "image": relative_target_path,  # Target: furnished room (what to generate)
                    "conditioning_image": relative_conditioning_path,  # Guidance: seg/depth of empty room
                    "input_image": relative_input_path  # Source: empty room (for reference)
                }
                
                writer.write(json.dumps(json_object) + '\n')
                processed_count += 1
            else:
                print(f"⚠️  Skipping {image_id}: Missing files")
                if not os.path.exists(input_image_path):
                    print(f"   Missing empty room: {input_image_path}")
                if not os.path.exists(conditioning_image_path):
                    print(f"   Missing conditioning: {conditioning_image_path}")
                if not os.path.exists(target_image_path):
                    print(f"   Missing furnished room: {target_image_path}")
                skipped_count += 1
    
    print(f"\n✅ Processing complete!")
    print(f"📝 Processed: {processed_count} image triplets")
    print(f"⏭️  Skipped: {skipped_count} image triplets")
    print(f"💾 Output saved to: {output_file}")
    print(f"🎯 Training: Empty room + {conditioning_type} + prompt → Furnished room")

def main():
    """Main function to create train.jsonl for both segmentation and depth conditioning."""
    
    # Define paths relative to the fill50k directory
    images_folder = "images"  # Empty rooms
    conditioning_images_folder = "conditioning_images"  # Seg/depth of empty rooms
    target_images_folder = "target_images"  # Furnished rooms
    captions_file = "../room_captions_robust.json"  # Captions describing furnished rooms
    
    # Create train.jsonl for segmentation conditioning
    print("🎨 Creating train.jsonl for SEGMENTATION conditioning...")
    print("   Empty room + segmentation → Furnished room")
    segmentation_output = "train_segmentation.jsonl"
    process_captions_json_corrected(captions_file, images_folder, conditioning_images_folder, 
                                  target_images_folder, segmentation_output, 
                                  conditioning_type="segmentation")
    
    print("\n" + "="*80 + "\n")
    
    # Create train.jsonl for depth conditioning  
    print("📐 Creating train.jsonl for DEPTH conditioning...")
    print("   Empty room + depth → Furnished room")
    depth_output = "train_depth.jsonl"
    process_captions_json_corrected(captions_file, images_folder, conditioning_images_folder, 
                                  target_images_folder, depth_output, 
                                  conditioning_type="depth")
    
    # Also create a default train.jsonl (using segmentation)
    print("\n📋 Creating default train.jsonl (segmentation)...")
    import shutil
    shutil.copy2(segmentation_output, "train.jsonl")
    print("✅ Default train.jsonl created (using segmentation conditioning)")
    
    print(f"\n🚀 Ready for training!")
    print(f"📁 Use dataset_name=fill50k.py in your training command")

if __name__ == "__main__":
    main() 