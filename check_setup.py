#!/usr/bin/env python3
"""
YOLO12 Setup Check - Simple Version

This script checks the YOLO12 setup without loading the full model.
"""

import os
import torch
import yaml


def main():
    """Main function."""
    print("=" * 60)
    print("YOLO12 Setup Check - No Training")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Device: {device}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    print("\n" + "-" * 40)
    
    # Check dataset configuration
    dataset_yaml = "dataset.yaml"
    if os.path.exists(dataset_yaml):
        print(f"Dataset config: {dataset_yaml} OK")
        
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Dataset path: {config['path']}")
        print(f"Number of classes: {config['nc']}")
        print(f"Class names: {list(config['names'].values())}")
    else:
        print(f"Dataset config: {dataset_yaml} ✗")
        return
    
    print("\n" + "-" * 40)
    
    # Check dataset structure
    yolo_dataset = "yolo_dataset"
    if os.path.exists(yolo_dataset):
        print(f"Dataset directory: {yolo_dataset} OK")
        
        # Count files in each split
        splits = ['train', 'val', 'test']
        total_images = 0
        
        for split in splits:
            images_dir = f"{yolo_dataset}/{split}/images"
            labels_dir = f"{yolo_dataset}/{split}/labels"
            
            if os.path.exists(images_dir):
                images = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                labels = len([f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]) if os.path.exists(labels_dir) else 0
                
                print(f"{split.capitalize()}: {images} images, {labels} labels")
                total_images += images
            else:
                print(f"{split.capitalize()}: Directory not found")
        
        print(f"Total images: {total_images}")
    else:
        print(f"Dataset directory: {yolo_dataset} ✗")
        return
    
    print("\n" + "-" * 40)
    
    # Check if YOLO12 model file exists
    model_file = "yolo12n.pt"
    if os.path.exists(model_file):
        print(f"YOLO12 model: {model_file} OK")
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        print(f"Model size: {file_size:.1f} MB")
    else:
        print(f"YOLO12 model: {model_file} NOT FOUND (will be downloaded when training starts)")
    
    print("\n" + "=" * 60)
    print("YOLO12 SETUP STATUS: READY")
    print("=" * 60)
    print("Environment is properly configured for YOLO12 training.")
    print("\nAvailable training options:")
    print("1. Quick training: python quick_start_yolo.py")
    print("2. Full training: python train_yolo12.py")
    print("3. Windows batch: start_yolo_training.bat")
    print("=" * 60)


if __name__ == "__main__":
    main()
