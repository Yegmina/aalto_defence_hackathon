#!/usr/bin/env python3
"""
Raspberry Pi Setup Script for Bottle Detection

This script helps set up the bottle detection system on Raspberry Pi 4.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_pi_environment():
    """Check if we're running on Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                return True
    except:
        pass
    return False


def install_requirements():
    """Install required packages for Raspberry Pi."""
    print("Installing requirements for Raspberry Pi...")
    
    # Update system
    subprocess.run(['sudo', 'apt', 'update'], check=True)
    subprocess.run(['sudo', 'apt', 'upgrade', '-y'], check=True)
    
    # Install system dependencies
    system_packages = [
        'python3-pip',
        'python3-venv',
        'python3-dev',
        'libhdf5-dev',
        'libhdf5-serial-dev',
        'libatlas-base-dev',
        'libjasper-dev',
        'libqtgui4',
        'libqt4-test',
        'python3-pyqt5',
        'libqtgui4',
        'libqt4-test',
        'python3-pyqt5',
        'libavcodec-dev',
        'libavformat-dev',
        'libswscale-dev',
        'libv4l-dev',
        'libxvidcore-dev',
        'libx264-dev',
        'libjpeg-dev',
        'libpng-dev',
        'libtiff-dev',
        'libatlas-base-dev',
        'gfortran',
        'cmake',
        'pkg-config'
    ]
    
    for package in system_packages:
        try:
            subprocess.run(['sudo', 'apt', 'install', '-y', package], check=True)
            print(f"Installed: {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install: {package}")
    
    # Install Python packages
    pip_packages = [
        'ultralytics',
        'opencv-python-headless',
        'numpy',
        'Pillow',
        'PyYAML',
        'matplotlib',
        'seaborn',
        'pandas',
        'tqdm',
        'psutil'
    ]
    
    for package in pip_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
            print(f"Installed Python package: {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install Python package: {package}")


def create_pi_optimized_script():
    """Create a Raspberry Pi optimized version of the detection script."""
    pi_script = '''#!/usr/bin/env python3
"""
Raspberry Pi Optimized Bottle Detection

This script is optimized for Raspberry Pi 4 with reduced resource usage.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import json
import time


def find_working_camera():
    """Find a working camera on Raspberry Pi."""
    print("Searching for working camera...")
    
    # Try different camera sources for Pi
    camera_sources = [0, 1, 2]  # USB cameras
    
    for camera_id in camera_sources:
        print(f"Trying camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera {camera_id} is working!")
                cap.release()
                return camera_id
            else:
                print(f"Camera {camera_id} opened but can't read frames")
        else:
            print(f"Camera {camera_id} not available")
        
        cap.release()
    
    print("No working camera found, using camera 0 as default")
    return 0


def main():
    """Main function for Pi-optimized bottle detection."""
    print("=" * 60)
    print("Raspberry Pi Bottle Detection")
    print("=" * 60)
    
    # Pi-optimized configuration
    width = 320  # Reduced resolution for better performance
    height = 240
    node_id = 456  # Different node ID for Pi
    
    # Detection thresholds
    CONF_THRESHOLD = 0.3  # Higher threshold for better performance
    IOU_THRESHOLD = 0.5
    
    # Allowed labels for detection
    allowed_labels = ['bottle', 'cup']
    
    print("Loading YOLO model...")
    try:
        # Use YOLO12n for better performance on Pi
        model = YOLO('yolo12n.pt')
    except Exception as e:
        print(f"Error loading YOLO12n: {e}")
        print("Falling back to YOLO11n...")
        try:
            model = YOLO('yolo11n.pt')
        except Exception as e2:
            print(f"Error loading YOLO11n: {e2}")
            print("Please install YOLO model first")
            return
    
    # Find working camera
    camera_id = find_working_camera()
    
    # Setup camera
    print(f"Setting up camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for Pi
    
    # Create output directory
    output_dir = 'pi_detection_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get the application start time
    start_time = datetime.now().strftime("%Y-%m-%d_%H.%M")
    sequence = 0
    
    # Create a black mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Statistics
    total_detections = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    print("Camera opened successfully!")
    print("Controls:")
    print("  'q' - quit")
    print("  's' - save screenshot")
    print("  'c' - switch camera")
    print(f"Detection threshold: {CONF_THRESHOLD}")
    print(f"Detection enabled for: {allowed_labels}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run detection (without tracking for better performance on Pi)
            results = model(
                frame,
                classes=[i for i, name in model.names.items() if name in allowed_labels],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )
            
            # Clear the mask
            mask[:] = 0
            
            # Process detections
            if results[0].boxes is not None:
                # Get the bounding boxes
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                target_count = len(boxes)
                total_detections += target_count
                
                # Data for JSON
                bounding_box_data = []
                
                # For each bounding box, copy the corresponding region from the original frame to the mask
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    confidence = confidences[i]
                    class_id = class_ids[i]
                    class_name = model.names[class_id]
                    
                    # Copy region to mask
                    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    
                    # Add to bounding box data
                    bounding_box_data.append({
                        "box_points": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(confidence),
                        "class_id": int(class_id),
                        "class_name": class_name
                    })
                
                # Save the mask and JSON data
                if target_count > 0:
                    sequence += 1
                    filename_base = f"pi_{start_time}_{sequence:05d}"
                    image_filename = f"{filename_base}.jpg"
                    json_filename = f"{filename_base}.json"
                    
                    image_filepath = os.path.join(output_dir, image_filename)
                    json_filepath = os.path.join(output_dir, json_filename)
                    
                    # Save mask image
                    cv2.imwrite(image_filepath, mask)
                    
                    # JSON output
                    output_data = {
                        "node_id": node_id,
                        "timestamp": datetime.now().isoformat(),
                        "sequence": sequence,
                        "target_count": target_count,
                        "bounding_box_data": bounding_box_data
                    }
                    
                    with open(json_filepath, 'w') as f:
                        json.dump(output_data, f, indent=4)
                    
                    print(f"Saved detection {sequence}: {target_count} targets")
            
            # Draw annotations on the original frame for display
            annotated_frame = results[0].plot()
            
            # Add statistics overlay
            cv2.putText(annotated_frame, f"Targets: {target_count}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Total: {total_detections}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(annotated_frame, f"Sequence: {sequence}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Camera: {camera_id}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display instructions
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", 
                       (10, annotated_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Show both frames
            cv2.imshow('Pi Bottle Detection', annotated_frame)
            cv2.imshow('Detection Mask', mask)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pi_detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                # Switch to next camera
                cap.release()
                camera_id = (camera_id + 1) % 3
                print(f"Switching to camera {camera_id}...")
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"Camera {camera_id} not available, switching back...")
                    camera_id = 0
                    cap = cv2.VideoCapture(camera_id)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, 15)
    
    except KeyboardInterrupt:
        print("\\nDetection stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\\nDetection completed!")
        print(f"Total detections: {total_detections}")
        print(f"Sequence count: {sequence}")
        print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
'''
    
    with open('pi_bottle_detection.py', 'w') as f:
        f.write(pi_script)
    
    print("Created pi_bottle_detection.py")


def create_transfer_script():
    """Create a script to transfer files to Raspberry Pi."""
    transfer_script = '''#!/bin/bash
# Transfer script for Raspberry Pi

echo "Transferring files to Raspberry Pi..."

# Replace with your Pi's IP address or hostname
PI_HOST="raspberrypi.local"
PI_USER="pi"
PI_PATH="/home/pi/bottle_detection"

# Create directory on Pi
ssh $PI_USER@$PI_HOST "mkdir -p $PI_PATH"

# Transfer files
scp enhanced_bottle_detection.py $PI_USER@$PI_HOST:$PI_PATH/
scp pi_bottle_detection.py $PI_USER@$PI_HOST:$PI_PATH/
scp setup_raspberry_pi.py $PI_USER@$PI_HOST:$PI_PATH/
scp requirements_yolo.txt $PI_USER@$PI_HOST:$PI_PATH/
scp dataset.yaml $PI_USER@$PI_HOST:$PI_PATH/

# Transfer YOLO model (if exists)
if [ -f "yolo12n.pt" ]; then
    scp yolo12n.pt $PI_USER@$PI_HOST:$PI_PATH/
fi

echo "Files transferred successfully!"
echo "Now SSH into your Pi and run:"
echo "  cd $PI_PATH"
echo "  python3 setup_raspberry_pi.py"
'''
    
    with open('transfer_to_pi.sh', 'w') as f:
        f.write(transfer_script)
    
    # Make executable
    os.chmod('transfer_to_pi.sh', 0o755)
    print("Created transfer_to_pi.sh")


def main():
    """Main setup function."""
    print("=" * 60)
    print("Raspberry Pi Setup for Bottle Detection")
    print("=" * 60)
    
    if check_pi_environment():
        print("Running on Raspberry Pi - installing requirements...")
        install_requirements()
        create_pi_optimized_script()
    else:
        print("Running on desktop - creating transfer files...")
        create_pi_optimized_script()
        create_transfer_script()
        
        print("\\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print("To transfer to Raspberry Pi:")
        print("1. Make sure your Pi is connected to the same network")
        print("2. Enable SSH on your Pi: sudo systemctl enable ssh")
        print("3. Run: ./transfer_to_pi.sh")
        print("4. SSH into your Pi and run: python3 setup_raspberry_pi.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
