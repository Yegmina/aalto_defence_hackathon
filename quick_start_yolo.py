#!/usr/bin/env python3
"""
Quick Start YOLO12 Script

This script provides a quick way to start YOLO12 training with default settings.
"""

import os
import sys
from ultralytics import YOLO
import torch


def check_environment():
    """Check the training environment."""
    print("=" * 60)
    print("YOLO12 Environment Check")
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
    else:
        print(f"Dataset directory: {yolo_dataset} ✗")
        return False
    
    print("=" * 60)
    return True


def quick_train():
    """Quick training with default settings."""
    print("Starting YOLO12 Quick Training...")
    
    # Load YOLO12 model
    model = YOLO('yolo12n.pt')  # Nano version for quick training
    
    # Train with optimized settings for quick results
    results = model.train(
        data='dataset.yaml',
        epochs=50,                    # Reduced epochs for quick training
        imgsz=640,
        batch=8,                      # Smaller batch size
        device='auto',                # Auto-detect device
        workers=2,                    # Fewer workers
        project='runs/train',
        name='yolo12_quick',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        patience=10,                  # Early stopping
        save_period=10,               # Save every 10 epochs
        cache=False,                  # Don't cache images
        rect=False,                   # Don't use rectangular training
        cos_lr=False,                 # Don't use cosine LR scheduler
        close_mosaic=10,              # Close mosaic augmentation in last 10 epochs
        resume=False,
        amp=True,                     # Use automatic mixed precision
        fraction=1.0,                 # Use full dataset
        profile=False,                # Don't profile
        freeze=None,                  # Don't freeze layers
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split='val',
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        plots=True,
        source=None,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        embed=None,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=None,
        format='torchscript',
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=False,
        opset=None,
        workspace=4,
        nms=False,
    )
    
    return results


def main():
    """Main function."""
    print("YOLO12 Quick Start for Defence Hackathon")
    
    # Check environment
    if not check_environment():
        print("Environment check failed. Please run setup_yolo12.py first.")
        sys.exit(1)
    
    # Ask user if they want to proceed
    response = input("\\nProceed with quick training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled.")
        return
    
    try:
        # Start training
        results = quick_train()
        
        print("\\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Results saved to: {results.save_dir}")
        print(f"Best mAP50: {results.best_fitness:.4f}")
        print(f"Best epoch: {results.best_epoch}")
        
        # Validate the model
        print("\\nRunning validation...")
        model = YOLO(results.save_dir + '/weights/best.pt')
        metrics = model.val()
        print(f"Validation mAP50: {metrics.box.map50:.4f}")
        print(f"Validation mAP50-95: {metrics.box.map:.4f}")
        
        print("\\nTraining completed successfully!")
        print("Check the runs/train/yolo12_quick/ directory for results.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
