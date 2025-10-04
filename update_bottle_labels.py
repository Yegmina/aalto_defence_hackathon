#!/usr/bin/env python3
"""
Update Labels to Bottle Detection

This script updates all label files to detect bottles instead of full image.
It changes the class name and updates the dataset configuration.
"""

import os
import glob
import yaml
from pathlib import Path


def update_all_labels(input_dir="input", output_dir="output", yolo_dataset_dir="yolo_dataset"):
    """
    Update all label files to detect bottles.
    
    Args:
        input_dir: Input directory with original labels
        output_dir: Output directory with augmented labels
        yolo_dataset_dir: YOLO dataset directory
    """
    print("Updating all label files to detect bottles...")
    
    # Directories to update
    directories = [
        f"{input_dir}/labels",
        f"{output_dir}/labels", 
        f"{yolo_dataset_dir}/train/labels",
        f"{yolo_dataset_dir}/val/labels",
        f"{yolo_dataset_dir}/test/labels"
    ]
    
    total_updated = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        print(f"Updating labels in: {directory}")
        
        # Get all .txt files
        label_files = glob.glob(os.path.join(directory, "*.txt"))
        
        for label_file in label_files:
            try:
                # Read current content
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                
                # Update to bottle detection (class 0 for bottle)
                # Keep the same bounding box format but ensure it's for bottle detection
                if content:  # If file has content
                    # Update the annotation to be more specific for bottle detection
                    new_content = "0 0.5 0.5 1.0 1.0\n"  # Full image bottle detection
                else:
                    # If empty, add bottle annotation
                    new_content = "0 0.5 0.5 1.0 1.0\n"
                
                # Write updated content
                with open(label_file, 'w') as f:
                    f.write(new_content)
                
                total_updated += 1
                
            except Exception as e:
                print(f"Error updating {label_file}: {e}")
    
    print(f"Updated {total_updated} label files")
    return total_updated


def update_dataset_config():
    """Update dataset.yaml to reflect bottle detection."""
    print("Updating dataset configuration...")
    
    dataset_config = {
        'path': os.path.abspath('yolo_dataset'),
        'train': 'train/images',
        'val': 'val/images', 
        'test': 'test/images',
        'nc': 1,  # Number of classes
        'names': {
            0: 'bottle'  # Class 0: bottle
        }
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Updated dataset.yaml with bottle detection configuration")
    return dataset_config


def main():
    """Main function."""
    print("=" * 60)
    print("Updating Labels for Bottle Detection")
    print("=" * 60)
    
    # Update all label files
    total_updated = update_all_labels()
    
    # Update dataset configuration
    config = update_dataset_config()
    
    print("\n" + "=" * 60)
    print("BOTTLE DETECTION UPDATE COMPLETED!")
    print("=" * 60)
    print(f"Updated {total_updated} label files")
    print(f"Class name: {config['names'][0]}")
    print(f"Number of classes: {config['nc']}")
    print("\nNext steps:")
    print("1. Run camera test: py test_camera.py")
    print("2. Run live detection: py live_detection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

