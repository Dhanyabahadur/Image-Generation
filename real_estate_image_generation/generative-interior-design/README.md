# Virtual Staging: AI-Powered Interior Design Generation

## Overview

This repository implements a comprehensive virtual staging system that generates furnished interior designs from empty rooms using text descriptions. The approach combines dual ControlNet architectures with advanced diffusion models to maintain architectural consistency while generating contextually appropriate furnishings and decor.

**Problem Statement**: Given an input empty room image and descriptive text, generate a furnished interior that accurately reflects the room description while preserving architectural layout and spatial relationships.

## Key Features

- **Dual ControlNet Architecture**: Segmentation and depth-based conditioning for enhanced layout preservation
- **Automated Dataset Pipeline**: Complete workflow from data collection to model training
- **Inpainting Integration**: Advanced empty room generation from furnished spaces
- **Multi-Modal Conditioning**: Text, segmentation, and depth-guided generation

## Technical Architecture

### Model Components
- **Segmentation ControlNet**: Preserves structural information (walls, windows, doors, floors)
- **Depth ControlNet**: Maintains spatial relationships and room geometry
- **Base Diffusion Model**: Stable Diffusion v1.5 for image generation
- **Caption Model**: LLaVa-1.5 for automated image description generation

### Pipeline Overview
1. **Data Collection**: Airbnb image scraping with global geographical distribution
2. **Content Filtering**: Indoor/outdoor classification and bathroom exclusion
3. **Feature Extraction**: Segmentation masks, depth maps, and empty room generation
4. **Caption Generation**: Automated description creation for training data
5. **Model Training**: Dual ControlNet and LoRA fine-tuning workflows

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (minimum 8GB VRAM recommended)
- PyTorch with CUDA support
- Accelerate library for distributed training

### Environment Setup

```bash
# Clone repository
git clone [your-repository-url]
cd generative-interior-design

# Create conda environment
conda create -n virtual-staging python=3.8
conda activate virtual-staging

# Install dependencies
pip install -r requirements.txt
```

## Dataset Creation

This pipeline extends the [bnb-dataset](https://github.com/airbert-vln/bnb-dataset/tree/main) codebase for comprehensive data collection and processing.

### 1. Data Collection

**Download Airbnb Listings**:
```bash
python search_listings_with_price.py --location data/cities_world.txt
```

**Extract Image Metadata**:
```bash
python download_listings.py --listings data/listings --output data/merlin --with_photo --num_splits 1 --start 0
```

**Create Unified TSV File**:
```bash
python extract_photo_metadata.py --merlin data/merlin/ --output data/bnb-dataset-raw.tsv
```

### 2. Image Processing

**Download and Preprocess Images** (768px resolution):
```bash
python download_images.py \
    --csv_file data/bnb-dataset-raw.tsv \
    --output data/images \
    --correspondance /tmp/cache-download-images/ \
    --num_splits 4 \
    --num_procs 4
```

### 3. Content Filtering

**Indoor/Outdoor Classification**:
```bash
python detect_room.py \
    --output data/places365/detection-results.tsv \
    --images data/images
```

**Extract Indoor Images Only**:
```bash
python extract_indoor.py \
    --output data/bnb-dataset-indoor.tsv \
    --detection data/places365/detection-results.tsv
```

### 4. Empty Room Generation

**Remove Furniture and Generate Conditioning Data**:
```bash
python get_empty_room.py
```

This script generates three outputs per input image:
- **Empty room RGB image**: Furniture removed via inpainting
- **Segmentation mask**: Structural element identification
- **Depth estimation**: Spatial relationship mapping

### 5. Dataset Preparation

**Generate Captions with LLaVa-1.5**:
```bash
python get_captions.py
```

**Create Training JSONL**:
```bash
python prepare_train_jsonl.py
```

### Required Directory Structure
```
dataset/
├── images/                 # Original furnished images
├── conditioning_images/    # Segmentation masks or depth maps
├── empty_rooms/           # Generated empty room images
├── train.jsonl           # Training metadata
├── fill50k.py           # Custom dataset loader
└── validation_images/    # Sample images for validation
```

## Model Training

### Segmentation ControlNet

```bash
# Clear cache and configure accelerate
rm -rf ~/.cache/huggingface/datasets/
accelerate config

# Train segmentation-conditioned ControlNet
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --controlnet_model_name_or_path="BertChristiaens/controlnet-seg-room" \
    --output_dir=models/controlnet-segmentation \
    --dataset_name=fill50k.py \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --checkpointing_steps=5000 \
    --validation_steps=2500 \
    --validation_image="./validation_images/room_1.jpg" "./validation_images/room_2.jpg" \
    --validation_prompt="Modern minimalist living room with clean lines and neutral colors" "Cozy bedroom with warm lighting and comfortable furnishings" \
    --mixed_precision="bf16" \
    --report_to=wandb
```

### Depth ControlNet

```bash
# Train depth-conditioned ControlNet
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_controlnet.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --controlnet_model_name_or_path="lllyasviel/sd-controlnet-depth" \
    --output_dir=models/controlnet-depth \
    --dataset_name=fill50k.py \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --checkpointing_steps=5000 \
    --validation_steps=2500 \
    --validation_image="./validation_images/room_1.jpg" "./validation_images/room_2.jpg" \
    --validation_prompt="Elegant dining room with sophisticated furniture arrangement" "Contemporary office space with modern productivity features" \
    --mixed_precision="bf16" \
    --report_to=wandb
```

### LoRA Fine-tuning

```bash
# Train LoRA adapter for style consistency
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir=models/lora-interior-design \
    --dataset_name=fill50k.py \
    --resolution=512 \
    --learning_rate=1e-4 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --rank=64 \
    --checkpointing_steps=5000 \
    --validation_prompt="Luxurious master bedroom with premium materials and sophisticated design elements" \
    --random_flip \
    --mixed_precision="bf16" \
    --report_to=wandb
```

## Results and Performance

### Dataset Characteristics
- **Source**: ~70,000 Airbnb images globally
- **Filtered Dataset**: 70 high-quality indoor images (constrained by storage/time)
- **Coverage**: Global geographical distribution with premium listing focus
- **Quality**: Moderate cleaning applied within time constraints

### Model Performance
- **Architectural Preservation**: Dual ControlNet approach maintains room layout effectively
- **Style Consistency**: LoRA fine-tuning improves prompt adherence
- **Generation Quality**: Output quality correlates with input image resolution and dataset quality

### Limitations and Challenges
- **Inpainting Consistency**: Generative empty room creation shows variable results
- **Dataset Size**: Limited training data affects model generalization
- **Computational Requirements**: Dual ControlNet inference requires substantial GPU memory

## Acknowledgments

This implementation builds upon several key contributions:
- **Dataset Foundation**: Extended from [bnb-dataset](https://github.com/airbert-vln/bnb-dataset/tree/main) methodology
- **AI Crowd**: Methodological inspiration from virtual staging challenges
- **Diffusers Library**: Training scripts and pipeline implementations
- **Research Community**: Foundational work in ControlNet and diffusion model architectures