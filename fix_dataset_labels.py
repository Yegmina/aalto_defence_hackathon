#!/usr/bin/env python3
"""
Fix Dataset Labels

This script fixes the class labels in the dataset to properly distinguish between bottles, cups, and cans.
Based on the detection results, we need to correct the class assignments.
"""

import os
import glob
import shutil
from pathlib import Path


def fix_labels_in_directory(labels_dir):
    """
    Fix labels in a directory by swapping class IDs.
    
    Args:
        labels_dir: Directory containing label files
    """
    print(f"Fixing labels in: {labels_dir}")
    
    if not os.path.exists(labels_dir):
        print(f"Directory not found: {labels_dir}")
        return 0
    
    fixed_count = 0
    
    for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        try:
            # Read current content
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Fix each line
            fixed_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Swap class IDs: 0->1 (bottle->cup), 1->0 (cup->bottle)
                        # This should fix the mismatch where cups were labeled as bottles
                        new_class_id = 1 - class_id
                        parts[0] = str(new_class_id)
                        fixed_lines.append(' '.join(parts) + '\n')
            
            # Write fixed content
            with open(label_file, 'w') as f:
                f.writelines(fixed_lines)
            
            fixed_count += 1
            
        except Exception as e:
            print(f"Error fixing {label_file}: {e}")
    
    print(f"Fixed {fixed_count} label files")
    return fixed_count


def main():
    """Main function to fix all labels."""
    print("=" * 60)
    print("Fixing Dataset Labels")
    print("=" * 60)
    
    # Directories to fix
    directories = [
        "new_dataset/input/labels",
        "new_dataset/output/labels",
        "new_dataset/yolo_dataset/train/labels",
        "new_dataset/yolo_dataset/val/labels",
        "new_dataset/yolo_dataset/test/labels"
    ]
    
    # Also fix augmented labels if they exist
    if os.path.exists("new_dataset/output/labels"):
        directories.append("new_dataset/output/labels")
    
    total_fixed = 0
    
    for directory in directories:
        if os.path.exists(directory):
            fixed = fix_labels_in_directory(directory)
            total_fixed += fixed
        else:
            print(f"Directory not found: {directory}")
    
    print("\n" + "=" * 60)
    print("LABEL FIXING COMPLETED!")
    print("=" * 60)
    print(f"Total label files fixed: {total_fixed}")
    print("\nNext steps:")
    print("1. Retrain model: py train_new_yolo12.py")
    print("2. Test detection: py realtime_new_detection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
