#!/usr/bin/env python3
"""
Fix Corrupted Labels

This script fixes the corrupted label files that have \\n characters in the middle of values.
"""

import os
import glob
import re


def fix_corrupted_label_file(label_file):
    """
    Fix a single corrupted label file.
    
    Args:
        label_file: Path to the label file to fix
    """
    try:
        # Read the file content
        with open(label_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the corrupted format by removing \\n and fixing the structure
        # The issue is that values are split across lines incorrectly
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove any \\n characters that shouldn't be there
                line = line.replace('\\n', '')
                
                # Check if this looks like a valid YOLO annotation
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Validate that all parts are valid numbers
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Ensure values are in valid range
                        if (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 <= width <= 1 and 0 <= height <= 1):
                            fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    except ValueError:
                        # Skip invalid lines
                        continue
        
        # Write the fixed content
        with open(label_file, 'w', encoding='utf-8') as f:
            for line in fixed_lines:
                f.write(line + '\n')
        
        return len(fixed_lines)
        
    except Exception as e:
        print(f"Error fixing {label_file}: {e}")
        return 0


def main():
    """Main function to fix all corrupted labels."""
    print("=" * 60)
    print("Fixing Corrupted Label Files")
    print("=" * 60)
    
    # Directories to fix
    directories = [
        "new_dataset/input/labels",
        "new_dataset/output/labels",
        "new_dataset/yolo_dataset/train/labels",
        "new_dataset/yolo_dataset/val/labels",
        "new_dataset/yolo_dataset/test/labels"
    ]
    
    total_fixed = 0
    total_files = 0
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nFixing labels in: {directory}")
            label_files = glob.glob(os.path.join(directory, "*.txt"))
            
            for label_file in label_files:
                annotations_count = fix_corrupted_label_file(label_file)
                if annotations_count > 0:
                    total_fixed += annotations_count
                    total_files += 1
                    print(f"  Fixed: {os.path.basename(label_file)} ({annotations_count} annotations)")
        else:
            print(f"Directory not found: {directory}")
    
    print("\n" + "=" * 60)
    print("LABEL FIXING COMPLETED!")
    print("=" * 60)
    print(f"Total files fixed: {total_files}")
    print(f"Total annotations fixed: {total_fixed}")
    print("\nNext steps:")
    print("1. Retrain model: py train_new_yolo12.py")
    print("2. Test detection: py realtime_new_detection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

