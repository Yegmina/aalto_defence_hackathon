#!/usr/bin/env python3
"""
Setup New Dataset Pipeline

This script sets up the complete pipeline for the new dataset from test2 folder:
1. Organize dataset structure
2. Create augmentation pipeline
3. Set up YOLO12 training
4. Create real-time detection
"""

import os
import shutil
import random
import yaml
from pathlib import Path
import glob


def setup_dataset_structure():
    """Set up the proper dataset structure for the new dataset."""
    print("Setting up dataset structure...")
    
    # Create directories
    directories = [
        "new_dataset/input/images",
        "new_dataset/input/labels",
        "new_dataset/output/images", 
        "new_dataset/output/labels",
        "new_dataset/yolo_dataset/train/images",
        "new_dataset/yolo_dataset/train/labels",
        "new_dataset/yolo_dataset/val/images",
        "new_dataset/yolo_dataset/val/labels",
        "new_dataset/yolo_dataset/test/images",
        "new_dataset/yolo_dataset/test/labels"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    return directories


def copy_new_dataset():
    """Copy the new dataset from test2 folder to proper structure."""
    print("Copying new dataset...")
    
    # Copy images
    source_images = "test2/extracted_photos"
    dest_images = "new_dataset/input/images"
    
    if os.path.exists(source_images):
        image_files = glob.glob(os.path.join(source_images, "*.jpg"))
        for image_file in image_files:
            shutil.copy2(image_file, dest_images)
        print(f"Copied {len(image_files)} images")
    else:
        print(f"Source images directory not found: {source_images}")
        return False
    
    # Copy labels
    source_labels = "test2/extracted_labels"
    dest_labels = "new_dataset/input/labels"
    
    if os.path.exists(source_labels):
        label_files = glob.glob(os.path.join(source_labels, "*.txt"))
        for label_file in label_files:
            shutil.copy2(label_file, dest_labels)
        print(f"Copied {len(label_files)} labels")
    else:
        print(f"Source labels directory not found: {source_labels}")
        return False
    
    return True


def analyze_dataset():
    """Analyze the dataset to understand classes and structure."""
    print("Analyzing dataset...")
    
    labels_dir = "new_dataset/input/labels"
    class_counts = {}
    total_objects = 0
    
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
    
    print(f"Total objects: {total_objects}")
    print(f"Classes found: {class_counts}")
    
    # Determine class names (you can modify these based on your dataset)
    class_names = {}
    for class_id in sorted(class_counts.keys()):
        if class_id == 0:
            class_names[class_id] = "bottle"
        elif class_id == 1:
            class_names[class_id] = "cup"
        else:
            class_names[class_id] = f"class_{class_id}"
    
    print(f"Class names: {class_names}")
    return class_names, class_counts


def split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split the dataset into train, validation, and test sets."""
    print(f"Splitting dataset: {train_ratio*100:.0f}% train, {val_ratio*100:.0f}% val, {test_ratio*100:.0f}% test")
    
    images_dir = "new_dataset/input/images"
    labels_dir = "new_dataset/input/labels"
    
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
            dst_image = f"new_dataset/yolo_dataset/{split_name}/images/{file}"
            shutil.copy2(src_image, dst_image)
            
            # Copy corresponding label
            base_name = Path(file).stem
            src_label = os.path.join(labels_dir, f"{base_name}.txt")
            dst_label = f"new_dataset/yolo_dataset/{split_name}/labels/{base_name}.txt"
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    print("Dataset split completed successfully!")
    return len(train_files), len(val_files), len(test_files)


def create_dataset_yaml(class_names):
    """Create the dataset configuration YAML file for YOLO."""
    print("Creating dataset configuration...")
    
    dataset_config = {
        'path': os.path.abspath('new_dataset/yolo_dataset'),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open('new_dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Created new_dataset.yaml configuration file")
    return dataset_config


def main():
    """Main function to set up the complete pipeline."""
    print("=" * 60)
    print("New Dataset Pipeline Setup")
    print("=" * 60)
    
    # Step 1: Setup dataset structure
    setup_dataset_structure()
    
    # Step 2: Copy new dataset
    if not copy_new_dataset():
        print("Error: Failed to copy dataset")
        return
    
    # Step 3: Analyze dataset
    class_names, class_counts = analyze_dataset()
    
    # Step 4: Split dataset
    train_count, val_count, test_count = split_dataset()
    
    # Step 5: Create dataset configuration
    dataset_config = create_dataset_yaml(class_names)
    
    print("\n" + "=" * 60)
    print("NEW DATASET SETUP COMPLETED!")
    print("=" * 60)
    print(f"Dataset structure created with:")
    print(f"  - Training images: {train_count}")
    print(f"  - Validation images: {val_count}")
    print(f"  - Test images: {test_count}")
    print(f"  - Total classes: {dataset_config['nc']}")
    print(f"  - Class names: {list(dataset_config['names'].values())}")
    print(f"  - Class distribution: {class_counts}")
    print("\nNext steps:")
    print("1. Run augmentation: python augment_new_dataset.py")
    print("2. Train YOLO12: python train_new_yolo12.py")
    print("3. Run detection: python realtime_new_detection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

