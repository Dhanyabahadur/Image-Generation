# Low-Light Image Enhancement via CLIP-Fourier Guided Wavelet Diffusion

## Overview

This repository contains an implementation of the CLIP-Fourier Guided Wavelet Diffusion (CFWD) model for low-light image enhancement. The approach leverages wavelet-based diffusion models that operate efficiently by transforming images from pixel space to wavelet space, enabling sophisticated enhancement techniques for low-light scenarios.

![](./Figs/fig2.png)

## Key Features

- **Wavelet-based Enhancement**: Utilizes wavelet space transformations for efficient image processing
- **CLIP-Fourier Guidance**: Incorporates CLIP features and Fourier domain processing for improved results
- **Pretrained Model Support**: Includes inference capabilities with pretrained models
- **Flexible Pipeline**: Modular design allowing for custom dataset integration

## Project Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the Repository**

2. **Create Environment and Install Dependencies**
   ```bash
   conda create -n cfwd python=3.8
   conda activate cfwd
   pip install -r requirements.txt
   ```

## Dataset and Models

### Training and Evaluation Datasets
Download the raw training and evaluation datasets from:
- **Dataset**: [[Google Drive]](https://drive.google.com/drive/folders/1yAp7c-fQhU_KQkK7xk1KZ4YKAywwo-2z?usp=drive_link)

### Pretrained Models
Download pretrained models and prompts from:
- **Models**: [[Google Drive]](https://drive.google.com/drive/folders/16tWuT7bVzQin2eiagsMByc-KN5UIQUho?usp=drive_link)

## Usage

### Quick Inference

1. **Download Pretrained Models**: Ensure you have downloaded the pretrained models (see above)

2. **Configure Environment**: Modify `test.py` and `datasets.py` according to your environment setup

3. **Run Inference**:
   ```bash
   python test.py
   ```

### Custom Dataset Integration

The pipeline supports custom datasets. Modify the dataset configuration in `datasets.py` to point to your data directory structure.

## Results

The model demonstrates effective enhancement of low-light images with:
- Improved illumination and detail preservation
- Maintained structural integrity
- Reduced noise artifacts

**Note**: Output quality correlates with input image quality. Higher resolution inputs yield better enhancement results.

## Technical Details

### Architecture
- **Wavelet Transform**: Efficient pixel-to-wavelet space conversion
- **Diffusion Process**: Guided denoising in wavelet domain
- **CLIP Integration**: Semantic guidance for enhancement quality
- **Fourier Components**: Frequency domain processing for detail preservation

### Performance Considerations
- GPU acceleration recommended for inference
- Memory requirements scale with input image resolution
- Batch processing supported for multiple images

## Future Improvements

- Fine-tuning on domain-specific datasets
- Integration with advanced diffusion architectures
- Real-time processing optimizations
- Enhanced prompt-based control mechanisms


**Original Paper**: [CFWD](https://arxiv.org/abs/2401.03788)

## Acknowledgments

This implementation builds upon the foundational work presented in the CFWD paper. We acknowledge the original authors for their contributions to the field of low-light image enhancement using wavelet diffusion models.
