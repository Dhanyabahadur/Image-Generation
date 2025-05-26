# from PIL import Image
# import os
# import json
# import time
# from transformers import BitsAndBytesConfig, pipeline
# import torch


# # Define constants
# IMAGES_FOLDER = '/mnt/data1/nick/depth/fill50k/images'
# IMAGES_FOLDER_2 = '/mnt/data1/nick/depth/fill50k/conditioning_images'
# OUTPUT_JSON_FILE = 'captions.json'
# BATCH_SIZE = 1000

# # Define prompt and model for image captioning
# prompt = "USER: <image>\nDescribe the interior image. Be detailed, describe a style, a color, and furniture fabric. Use only one but detailed sentence. It must begin with room type description, then always describe the general style and after that describe the all furniture items and their arrangement in the room and color. \n ASSISTANT:"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
# pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf", model_kwargs={"quantization_config": quantization_config})

# def generate_caption(image_path):
#     """Generate caption for the given image."""
#     image = Image.open(image_path)
#     outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 100, "temperature":0.4, "do_sample":True})
#     text = outputs[0]["generated_text"]
#     return text[text.find('ASSISTANT:')+len('ASSISTANT:')+1:]

# def filter_image_files(images_folder, other_folder):
#     """Filter image files based on existence of corresponding segmentation files."""
#     image_files = sorted(os.listdir(images_folder))
#     other_files = set(os.listdir(other_folder))
#     filtered_files = [filename for filename in image_files if f"{os.path.splitext(filename)[0]}segmentation.png" in other_files]
#     return filtered_files

# def process_images(images_folder, images_folder2, output_json_file, batch_size):
#     """Process images and generate captions."""
#     image_files = sorted(filter_image_files(images_folder, images_folder2))
#     captions_dict = {}
#     start_time = time.time()
#     for i, image_file in enumerate(image_files, start=1):
#         image_path = os.path.join(images_folder, image_file)
#         caption = generate_caption(image_path)
#         captions_dict[image_file] = caption
#         if i % batch_size == 0 or i == len(image_files):
#             with open(output_json_file, 'a') as json_file:
#                 json.dump(captions_dict, json_file, indent=4)
#             print(f"Processed {i}/{len(image_files)} images. Results dumped to {output_json_file}. Elapsed time: {time.time() - start_time:.2f} seconds")
#             captions_dict = {}
#             start_time = time.time()
#     print("All images processed and results dumped to", output_json_file)

# if __name__ == "__main__":
#     process_images(IMAGES_FOLDER, IMAGES_FOLDER_2, OUTPUT_JSON_FILE, BATCH_SIZE)


from PIL import Image
import os
import json
import time
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor
import torch

# Define constants
DATASET_TSV = "dataset_gathering/data/bnb-dataset-indoor-25koffers.tsv"
IMAGES_BASE_DIR = "dataset_gathering/data/images_price/"
CLEAN_IMAGES_DIR = "get_empty_room/clean_images/"
OUTPUT_JSON_FILE = 'room_captions_robust.json'
BATCH_SIZE = 10

# Setup model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading LLaVA model and processor...")
try:
    # Try the standard LLaVA model first
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", 
        quantization_config=quantization_config, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying alternative loading method...")
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    print("Alternative model loaded successfully!")

def find_image_path(image_id, base_dir):
    """Find the full path of an image given its ID and base directory"""
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            image_path = os.path.join(subdir_path, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
    return None

def generate_caption(image_path):
    """Generate caption for the given image using direct model approach."""
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Resize if needed
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Use chat format as suggested by the warning
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this interior room in detail, starting with room type, then style, furniture, colors, and arrangement."},
                    {"type": "image"}
                ]
            }
        ]
        
        # Apply chat template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        caption = processor.decode(generated_ids, skip_special_tokens=True)
        
        return caption.strip()
        
    except Exception as e:
        print(f"Error generating caption for {image_path}: {e}")
        # Fallback to simple prompt if chat format fails
        try:
            simple_prompt = "USER: <image>\nDescribe this interior room in detail.\nASSISTANT:"
            inputs = processor(simple_prompt, image, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            caption = processor.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in caption:
                caption = caption.split("ASSISTANT:")[-1].strip()
            
            return caption
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return ""

def load_and_filter_images():
    """Load and filter images"""
    # Load indoor IDs
    indoor_ids = []
    with open(DATASET_TSV, 'r') as f:
        for line in f:
            indoor_ids.append(line.strip())
    
    # Filter to successfully processed images
    if not os.path.exists(CLEAN_IMAGES_DIR):
        return []
    
    clean_files = set(os.listdir(CLEAN_IMAGES_DIR))
    processed_ids = []
    
    for image_id in indoor_ids:
        if f"{image_id}_clean.png" in clean_files:
            processed_ids.append(image_id)
    
    return processed_ids

def main():
    processed_ids = load_and_filter_images()
    print(f"Found {len(processed_ids)} images to process")
    
    captions_dict = {}
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            with open(OUTPUT_JSON_FILE, 'r') as f:
                captions_dict = json.load(f)
            print(f"Loaded {len(captions_dict)} existing captions")
        except:
            pass
    
    successful = 0
    failed = 0
    
    for i, image_id in enumerate(processed_ids, 1):
        if image_id in captions_dict:
            print(f"Skipping {i}/{len(processed_ids)}: {image_id} (already processed)")
            continue
            
        image_path = find_image_path(image_id, IMAGES_BASE_DIR)
        if not image_path:
            print(f"Image not found: {image_id}")
            failed += 1
            continue
            
        print(f"Processing {i}/{len(processed_ids)}: {image_id}")
        caption = generate_caption(image_path)
        
        if caption:
            captions_dict[image_id] = {
                "image_path": image_path,
                "caption": caption,
                "timestamp": time.time()
            }
            successful += 1
            print(f"  ✓ Caption: {caption}...")
        else:
            failed += 1
            print(f"  ✗ Failed to generate caption")
        
        # Save checkpoint every BATCH_SIZE images
        if i % BATCH_SIZE == 0:
            with open(OUTPUT_JSON_FILE, 'w') as f:
                json.dump(captions_dict, f, indent=2)
            print(f"Checkpoint saved: {successful} successful, {failed} failed")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Final save
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(captions_dict, f, indent=2)
    
    print(f"\n🎉 Captioning complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {OUTPUT_JSON_FILE}")

if __name__ == "__main__":
    main() 