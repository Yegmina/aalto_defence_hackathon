#!/usr/bin/env python3
"""
Augment New Dataset

This script performs data augmentation on the new dataset with proper YOLO annotations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def setup_augmentations() -> A.Compose:
    """
    Set up the augmentation pipeline using albumentations.
    
    Returns:
        A.Compose: Configured augmentation pipeline with YOLO bbox parameters
    """
    return A.Compose([
        # Geometric transformations
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=10,
            p=0.3
        ),
        
        # Photometric transformations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=25,
            val_shift_limit=15,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        
        # Noise and blur effects
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        
        # Color transformations
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomToneCurve(scale=0.1, p=0.3),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.1
    ))


def read_yolo_annotations(annotation_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Read YOLO format annotation file.
    
    Args:
        annotation_path: Path to the annotation file
        
    Returns:
        Tuple of (bounding_boxes, class_labels)
    """
    bounding_boxes = []
    class_labels = []
    
    if not os.path.exists(annotation_path):
        return bounding_boxes, class_labels
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            bounding_boxes.append([x_center, y_center, width, height])
                            class_labels.append(class_id)
                        except ValueError as ve:
                            print(f"Warning: Invalid format in {annotation_path}: {line} - {ve}")
                            continue
    except Exception as e:
        print(f"Warning: Could not read annotation file {annotation_path}: {e}")
    
    return bounding_boxes, class_labels


def write_yolo_annotations(annotation_path: str, bounding_boxes: List[List[float]], class_labels: List[int]) -> None:
    """
    Write YOLO format annotation file.
    
    Args:
        annotation_path: Path to save the annotation file
        bounding_boxes: List of bounding boxes in YOLO format
        class_labels: List of class IDs
    """
    try:
        with open(annotation_path, 'w') as f:
            for bbox, class_id in zip(bounding_boxes, class_labels):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    except Exception as e:
        print(f"Warning: Could not write annotation file {annotation_path}: {e}")


def process_single_image(image_path: str, label_path: str, output_images_dir: str, 
                        output_labels_dir: str, num_augmentations: int, 
                        augmentation_pipeline: A.Compose) -> int:
    """
    Process a single image and create augmented versions.
    
    Args:
        image_path: Path to the input image
        label_path: Path to the input label file
        output_images_dir: Directory to save augmented images
        output_labels_dir: Directory to save augmented labels
        num_augmentations: Number of augmented versions to create
        augmentation_pipeline: Albumentations pipeline
        
    Returns:
        Number of successful augmentations created
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return 0
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read annotations
    bounding_boxes, class_labels = read_yolo_annotations(label_path)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    successful_augmentations = 0
    
    for i in range(num_augmentations):
        try:
            # Apply augmentations
            augmented = augmentation_pipeline(
                image=image,
                bboxes=bounding_boxes,
                class_labels=class_labels
            )
            
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            augmented_labels = augmented['class_labels']
            
            # Save augmented image
            output_image_path = os.path.join(output_images_dir, f"{base_name}_aug_{i+1}.jpg")
            augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, augmented_image_bgr)
            
            # Save augmented annotations
            output_label_path = os.path.join(output_labels_dir, f"{base_name}_aug_{i+1}.txt")
            write_yolo_annotations(output_label_path, augmented_bboxes, augmented_labels)
            
            successful_augmentations += 1
            
        except Exception as e:
            print(f"Warning: Failed to augment {image_path} (attempt {i+1}): {e}")
            continue
    
    return successful_augmentations


def main():
    """Main function to run the data augmentation script."""
    parser = argparse.ArgumentParser(
        description="Augment new YOLO dataset with various transformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python augment_new_dataset.py --input_dir ./new_dataset/input --output_dir ./new_dataset/output --num_augmentations 5
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./new_dataset/input',
        help='Path to the source directory containing images and labels subfolders'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./new_dataset/output',
        help='Path to the directory where augmented data will be saved'
    )
    
    parser.add_argument(
        '--num_augmentations',
        type=int,
        default=5,
        help='Number of augmented versions to create for each original image (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Setup directories
    input_images_dir = os.path.join(args.input_dir, 'images')
    input_labels_dir = os.path.join(args.input_dir, 'labels')
    output_images_dir = os.path.join(args.output_dir, 'images')
    output_labels_dir = os.path.join(args.output_dir, 'labels')
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Check if input images directory exists
    if not os.path.exists(input_images_dir):
        print(f"Error: Images directory '{input_images_dir}' does not exist.")
        print("Please ensure your input directory has an 'images' subfolder.")
        sys.exit(1)
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for file in os.listdir(input_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print(f"Error: No image files found in '{input_images_dir}'")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images to process")
    print(f"Creating {args.num_augmentations} augmented versions for each image")
    print(f"Total augmented images to be created: {len(image_files) * args.num_augmentations}")
    print("-" * 50)
    
    # Setup augmentation pipeline
    augmentation_pipeline = setup_augmentations()
    
    # Process images
    total_successful = 0
    total_processed = 0
    
    with tqdm(total=len(image_files), desc="Processing images", unit="image") as pbar:
        for image_file in image_files:
            # Construct paths
            image_path = os.path.join(input_images_dir, image_file)
            base_name = Path(image_file).stem
            label_file = f"{base_name}.txt"
            label_path = os.path.join(input_labels_dir, label_file)
            
            # Process the image
            successful = process_single_image(
                image_path, label_path, output_images_dir, output_labels_dir,
                args.num_augmentations, augmentation_pipeline
            )
            
            total_successful += successful
            total_processed += 1
            pbar.update(1)
            pbar.set_postfix({
                'successful': successful,
                'total_augmented': total_successful
            })
    
    # Print completion summary
    print("\n" + "=" * 50)
    print("AUGMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Images processed: {total_processed}")
    print(f"Total augmented images created: {total_successful}")
    print(f"Expected augmented images: {total_processed * args.num_augmentations}")
    print(f"Success rate: {(total_successful / (total_processed * args.num_augmentations)) * 100:.1f}%")
    print(f"\nOutput saved to:")
    print(f"  Images: {output_images_dir}")
    print(f"  Labels: {output_labels_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
