import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from tqdm import tqdm

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    # needs to be updated model path after unzipping
    parser.add_argument('--resume', default='./pretrain_models/pretrain_model.pth', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='./result', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("./configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def ensure_val_dataset(config):
    # Create the result folder
    os.makedirs(os.path.join('./result', config.data.val_dataset), exist_ok=True)
    
    # Ensure val_data_val.txt exists and has content
    val_txt = './data/val_data_val.txt'
    if not os.path.exists(val_txt) or os.path.getsize(val_txt) == 0:
        print("Generating val_data_val.txt...")
        
        # Path to validation images
        val_path = './data/val_data/val/low'
        if not os.path.exists(val_path):
            print(f"Validation path {val_path} does not exist!")
            return False
        
        # Get all image files
        image_files = []
        for f in os.listdir(val_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.append(f)
        
        if not image_files:
            print(f"No image files found in {val_path}")
            return False
        
        # Create normal directory and copy files if needed
        normal_path = val_path.replace('low', 'normal')
        os.makedirs(normal_path, exist_ok=True)
        for img in image_files:
            if not os.path.exists(os.path.join(normal_path, img)):
                # Copy from low to normal as a placeholder
                try:
                    import shutil
                    shutil.copy(os.path.join(val_path, img), os.path.join(normal_path, img))
                except Exception as e:
                    print(f"Error copying file: {e}")
        
        # Write filenames to val_data_val.txt
        with open(val_txt, 'w') as f:
            for img in image_files:
                f.write(img + '\n')
        
        print(f"Generated val_data_val.txt with {len(image_files)} images")
    
    return True


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Current device: {}".format(device))
    if torch.cuda.is_available():
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print("Current GPU: {} - {}".format(current_gpu, gpu_name))
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # Ensure validation dataset is properly set up
    if not ensure_val_dataset(config):
        print("Failed to set up validation dataset")
        return

    
    # data loading
    print("Current dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    val_loader = DATASET.get_loaders()

    # Check if validation loader is empty
    val_size = len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 0
    print(f"Validation dataset size: {val_size}")
    if val_size == 0:
        print("Warning: Validation dataset is empty!")
        return

    
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    val_loader_with_progress = tqdm(val_loader, desc='Loading Validation Data', leave=False)
    model.restore(val_loader_with_progress)

if __name__ == '__main__':
    main()