#!/usr/bin/env python3
"""
YOLO12 Setup Only - No Training

This script sets up YOLO12 environment and shows configuration without starting training.
"""

import os
import torch
from ultralytics import YOLO
import yaml


def check_environment():
    """Check the training environment without starting training."""
    print("=" * 60)
    print("YOLO12 Environment Setup Check")
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
    
    # Check dataset
    dataset_yaml = "dataset.yaml"
    if os.path.exists(dataset_yaml):
        print(f"Dataset config: {dataset_yaml} ✓")
        
        # Read and display dataset config
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"Dataset path: {config['path']}")
        print(f"Number of classes: {config['nc']}")
        print(f"Class names: {list(config['names'].values())}")
    else:
        print(f"Dataset config: {dataset_yaml} ✗")
        return False
    
    # Check dataset structure
    yolo_dataset = "yolo_dataset"
    if os.path.exists(yolo_dataset):
        print(f"Dataset directory: {yolo_dataset} ✓")
        
        # Count files
        train_images = len([f for f in os.listdir(f"{yolo_dataset}/train/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        val_images = len([f for f in os.listdir(f"{yolo_dataset}/val/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_images = len([f for f in os.listdir(f"{yolo_dataset}/test/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Training images: {train_images}")
        print(f"Validation images: {val_images}")
        print(f"Test images: {test_images}")
        print(f"Total images: {train_images + val_images + test_images}")
    else:
        print(f"Dataset directory: {yolo_dataset} ✗")
        return False
    
    print("=" * 60)
    return True


def load_model_info():
    """Load YOLO12 model and show information without training."""
    print("Loading YOLO12 model information...")
    
    try:
        # Load YOLO12 model
        model = YOLO('yolo12n.pt')
        
        print("YOLO12 Model Information:")
        print(f"Model type: {model.model_name}")
        print(f"Model size: {model.model_size}")
        print(f"Number of parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        
        # Show model summary
        print("\nModel Architecture:")
        model.info(verbose=False)
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def main():
    """Main function."""
    print("YOLO12 Setup Check - No Training")
    print("This script only sets up and checks the environment.")
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please run setup_yolo12.py first.")
        return
    
    # Load model info
    if not load_model_info():
        print("Model loading failed.")
        return
    
    print("\n" + "=" * 60)
    print("YOLO12 SETUP COMPLETE!")
    print("=" * 60)
    print("Environment is ready for training.")
    print("To start training, run:")
    print("  python quick_start_yolo.py")
    print("  or")
    print("  python train_yolo12.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

