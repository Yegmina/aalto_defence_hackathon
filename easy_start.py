#!/usr/bin/env python3
"""
Easy Start Script for YOLO Data Augmentation

This script provides a simple interface to run data augmentation with default settings.
It automatically sets up the directory structure and runs the augmentation process.
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_directory_structure(input_dir: str) -> None:
    """
    Set up the required directory structure for YOLO format.
    Creates images and labels subdirectories if they don't exist.
    """
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    
    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"✓ Directory structure set up:")
    print(f"  - {images_dir}")
    print(f"  - {labels_dir}")


def move_images_to_images_folder(input_dir: str) -> None:
    """
    Move all image files from the input directory to the images subfolder.
    """
    images_dir = os.path.join(input_dir, 'images')
    moved_count = 0
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for file in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                src_path = os.path.join(input_dir, file)
                dst_path = os.path.join(images_dir, file)
                
                try:
                    os.rename(src_path, dst_path)
                    moved_count += 1
                    print(f"  Moved: {file}")
                except Exception as e:
                    print(f"  Warning: Could not move {file}: {e}")
    
    if moved_count > 0:
        print(f"✓ Moved {moved_count} images to images folder")
    else:
        print("ℹ No images found to move")


def create_sample_labels(input_dir: str) -> None:
    """
    Create sample label files for images that don't have corresponding labels.
    This creates empty label files to prevent errors during augmentation.
    """
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    
    created_count = 0
    
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']):
            base_name = Path(file).stem
            label_file = f"{base_name}.txt"
            label_path = os.path.join(labels_dir, label_file)
            
            if not os.path.exists(label_path):
                try:
                    with open(label_path, 'w') as f:
                        # Create empty label file
                        pass
                    created_count += 1
                except Exception as e:
                    print(f"  Warning: Could not create label file for {file}: {e}")
    
    if created_count > 0:
        print(f"✓ Created {created_count} empty label files")
    else:
        print("ℹ All images already have corresponding label files")


def run_augmentation(input_dir: str, output_dir: str, num_augmentations: int = 5) -> None:
    """
    Run the data augmentation script.
    """
    print(f"\n🚀 Starting data augmentation...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Augmentations per image: {num_augmentations}")
    print("-" * 50)
    
    try:
        # Run the augmentation script
        cmd = [
            sys.executable, 'augment_data.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--num_augmentations', str(num_augmentations)
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running augmentation: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: augment_data.py not found. Please ensure it's in the same directory.")
        sys.exit(1)


def main():
    """Main function for easy start."""
    print("=" * 60)
    print("🎯 YOLO Data Augmentation - Easy Start")
    print("=" * 60)
    
    # Default settings
    input_dir = "input"
    output_dir = "output"
    num_augmentations = 5
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' does not exist.")
        print("Please ensure you have an 'input' folder with images.")
        sys.exit(1)
    
    print(f"📁 Setting up directory structure...")
    setup_directory_structure(input_dir)
    
    print(f"\n📷 Organizing images...")
    move_images_to_images_folder(input_dir)
    
    print(f"\n📝 Creating sample labels...")
    create_sample_labels(input_dir)
    
    print(f"\n🔧 Running augmentation...")
    run_augmentation(input_dir, output_dir, num_augmentations)
    
    print(f"\n🎉 All done! Check the '{output_dir}' folder for your augmented dataset.")


if __name__ == "__main__":
    main()


