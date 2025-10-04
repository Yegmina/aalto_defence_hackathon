#!/usr/bin/env python3
"""
Test Bottle Detection

This script tests bottle detection on sample images and tries different camera sources.
"""

import cv2
import torch
from ultralytics import YOLO
import os
import glob


def test_on_images():
    """Test bottle detection on sample images."""
    print("Testing bottle detection on sample images...")
    
    # Load YOLO12 model
    model = YOLO('yolo12n.pt')
    
    # Test on input images
    image_dir = "input/images"
    if os.path.exists(image_dir):
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))[:5]  # Test first 5 images
        
        for image_file in image_files:
            print(f"Testing: {os.path.basename(image_file)}")
            
            # Read image
            image = cv2.imread(image_file)
            if image is None:
                continue
            
            # Run detection
            results = model(image, verbose=False)
            
            # Count bottles
            bottle_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Check if it's a bottle (class 39 in COCO dataset)
                        if class_id == 39 and confidence > 0.3:  # Bottle class
                            bottle_count += 1
                            print(f"  Bottle detected with confidence: {confidence:.2f}")
            
            print(f"  Total bottles found: {bottle_count}")
            
            # Draw results
            annotated_image = results[0].plot()
            
            # Add bottle counter
            cv2.putText(annotated_image, f"Bottles: {bottle_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save result
            output_file = f"bottle_test_{os.path.basename(image_file)}"
            cv2.imwrite(output_file, annotated_image)
            print(f"  Result saved: {output_file}")
            print()


def test_camera_sources():
    """Test different camera sources."""
    print("Testing different camera sources...")
    
    for camera_id in range(3):  # Try cameras 0, 1, 2
        print(f"Trying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {camera_id} is working!")
                cap.release()
                return camera_id
            else:
                print(f"Camera {camera_id} opened but can't read frames")
        else:
            print(f"Camera {camera_id} not available")
        
        cap.release()
    
    print("No working camera found")
    return None


def main():
    """Main function."""
    print("=" * 60)
    print("Bottle Detection Test")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test on images first
    test_on_images()
    
    # Test camera sources
    working_camera = test_camera_sources()
    
    if working_camera is not None:
        print(f"\\nWorking camera found: {working_camera}")
        print("You can now run: py bottle_detection.py")
    else:
        print("\\nNo camera available, but image detection is working!")
        print("Check the generated test images to see bottle detection results.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

