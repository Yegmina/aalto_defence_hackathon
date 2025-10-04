#!/usr/bin/env python3
"""
YOLO12 Setup Script

This script sets up YOLO12 (YOLOv12) training environment for the defence hackathon dataset.
It includes dataset preparation, configuration, and training setup.
"""

import os
import shutil
import random
from pathlib import Path
import yaml


def create_dataset_structure():
    """Create the proper YOLO dataset structure."""
    print("Creating YOLO dataset structure...")
    
    # Create directories
    directories = [
        "yolo_dataset/train/images",
        "yolo_dataset/train/labels", 
        "yolo_dataset/val/images",
        "yolo_dataset/val/labels",
        "yolo_dataset/test/images",
        "yolo_dataset/test/labels"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    return directories


def split_dataset(input_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        input_dir: Path to input directory with images and labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
    """
    print(f"Splitting dataset: {train_ratio*100:.0f}% train, {val_ratio*100:.0f}% val, {test_ratio*100:.0f}% test")
    
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")
    
    # Get all image files
    image_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    # Shuffle the files
    random.seed(42)  # For reproducible splits
    random.shuffle(image_files)
    
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Split files
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    print(f"Total files: {total_files}")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files") 
    print(f"Test: {len(test_files)} files")
    
    # Copy files to respective directories
    def copy_files(file_list, split_name):
        for file in file_list:
            # Copy image
            src_image = os.path.join(images_dir, file)
            dst_image = f"yolo_dataset/{split_name}/images/{file}"
            shutil.copy2(src_image, dst_image)
            
            # Copy corresponding label
            base_name = Path(file).stem
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            dst_label = f"yolo_dataset/{split_name}/labels/{base_name}.txt"
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print("Dataset split completed successfully!")
    return len(train_files), len(val_files), len(test_files)


def create_dataset_yaml():
    """Create the dataset configuration YAML file for YOLO."""
    print("Creating dataset configuration...")
    
    dataset_config = {
        'path': os.path.abspath('yolo_dataset'),  # Dataset root dir
        'train': 'train/images',  # Train images (relative to 'path')
        'val': 'val/images',      # Val images (relative to 'path')
        'test': 'test/images',    # Test images (relative to 'path')
        
        # Number of classes
        'nc': 1,
        
        # Class names
        'names': {
            0: 'target_object'  # Class 0: your target object
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Created dataset.yaml configuration file")
    return dataset_config


def create_training_script():
    """Create a training script for YOLO12."""
    training_script = '''#!/usr/bin/env python3
"""
YOLO12 Training Script

This script trains a YOLO12 model on the defence hackathon dataset.
"""

import torch
from ultralytics import YOLO
import os


def main():
    """Main training function."""
    print("=" * 60)
    print("YOLO12 Training for Defence Hackathon")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO12 model
    print("Loading YOLO12 model...")
    model = YOLO('yolo12n.pt')  # You can use yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data='dataset.yaml',           # Dataset configuration
        epochs=100,                    # Number of training epochs
        imgsz=640,                     # Input image size
        batch=16,                      # Batch size (adjust based on GPU memory)
        device=device,                 # Device to use
        workers=4,                     # Number of worker threads
        project='runs/train',          # Project directory
        name='yolo12_defence_hackathon',  # Experiment name
        exist_ok=True,                 # Overwrite existing experiment
        pretrained=True,               # Use pretrained weights
        optimizer='AdamW',             # Optimizer
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate (lr0 * lrf)
        momentum=0.937,                # SGD momentum
        weight_decay=0.0005,           # Optimizer weight decay
        warmup_epochs=3,               # Warmup epochs
        warmup_momentum=0.8,           # Warmup initial momentum
        warmup_bias_lr=0.1,            # Warmup initial bias lr
        box=7.5,                       # Box loss gain
        cls=0.5,                       # Cls loss gain
        dfl=1.5,                       # Dfl loss gain
        pose=12.0,                     # Pose loss gain
        kobj=1.0,                      # Keypoint obj loss gain
        label_smoothing=0.0,           # Label smoothing
        nbs=64,                        # Nominal batch size
        overlap_mask=True,             # Masks should overlap during training
        mask_ratio=4,                  # Mask downsample ratio
        dropout=0.0,                   # Use dropout regularization
        val=True,                      # Validate/test during training
        split='val',                   # Dataset split to use for validation
        save_json=False,               # Save results to JSON
        save_hybrid=False,             # Save hybrid version of labels
        conf=None,                     # Object confidence threshold
        iou=0.7,                       # NMS IoU threshold
        max_det=300,                   # Maximum detections per image
        half=False,                    # Use half precision (FP16)
        dnn=False,                     # Use OpenCV DNN for ONNX inference
        plots=True,                    # Save plots during training
        source=None,                   # Source directory for images or videos
        vid_stride=1,                  # Video frame-rate stride
        stream_buffer=False,           # Buffer all streaming images
        visualize=False,               # Visualize model features
        augment=False,                 # Apply image augmentation to prediction sources
        agnostic_nms=False,            # Class-agnostic NMS
        retina_masks=False,            # Use high-resolution segmentation masks
        embed=None,                    # Return feature vectors/embeddings from given layers
        show=False,                    # Show results if possible
        save_frames=False,             # Save video frames
        save_txt=False,                # Save results as .txt file
        save_conf=False,               # Save results with confidence scores
        save_crop=False,               # Save cropped images with results
        show_labels=True,              # Show object labels in plots
        show_conf=True,                # Show object confidence scores in plots
        show_boxes=True,               # Show object bounding boxes
        line_width=None,               # Line width of bounding boxes
        format='torchscript',          # Format to export to
        keras=False,                   # Use Keras
        optimize=False,                # TorchScript: optimize for mobile
        int8=False,                    # CoreML/TF INT8 quantization
        dynamic=False,                 # ONNX/TF/TensorRT: dynamic axes
        simplify=False,                # ONNX: simplify model
        opset=None,                    # ONNX: opset version
        workspace=4,                   # TensorRT: workspace size (GB)
        nms=False,                     # CoreML: add NMS
    )
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Print training results
    print("\\nTraining Results:")
    print(f"Best mAP50: {results.best_fitness:.4f}")
    print(f"Best epoch: {results.best_epoch}")
    
    # Validate the model
    print("\\nRunning validation...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
'''
    
    with open('train_yolo12.py', 'w') as f:
        f.write(training_script)
    
    print("Created train_yolo12.py training script")


def create_requirements_yolo():
    """Create requirements file for YOLO12."""
    requirements = '''# YOLO12 Requirements
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
Pillow>=9.5.0
numpy>=1.24.0
PyYAML>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
psutil>=5.9.0
thop>=0.1.0
'''
    
    with open('requirements_yolo.txt', 'w') as f:
        f.write(requirements)
    
    print("Created requirements_yolo.txt")


def main():
    """Main setup function."""
    print("=" * 60)
    print("YOLO12 Setup for Defence Hackathon")
    print("=" * 60)
    
    # Create dataset structure
    create_dataset_structure()
    
    # Split the dataset
    train_count, val_count, test_count = split_dataset("input")
    
    # Create dataset configuration
    dataset_config = create_dataset_yaml()
    
    # Create training script
    create_training_script()
    
    # Create requirements file
    create_requirements_yolo()
    
    print("\\n" + "=" * 60)
    print("YOLO12 SETUP COMPLETED!")
    print("=" * 60)
    print(f"Dataset structure created with:")
    print(f"  - Training images: {train_count}")
    print(f"  - Validation images: {val_count}")
    print(f"  - Test images: {test_count}")
    print(f"  - Total classes: {dataset_config['nc']}")
    print(f"  - Class names: {list(dataset_config['names'].values())}")
    print("\\nNext steps:")
    print("1. Install YOLO requirements: pip install -r requirements_yolo.txt")
    print("2. Run training: python train_yolo12.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
