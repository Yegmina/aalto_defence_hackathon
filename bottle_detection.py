#!/usr/bin/env python3
"""
Bottle Detection with Camera

This script performs real-time bottle detection using YOLO12 and your webcam.
"""

import cv2
import torch
from ultralytics import YOLO
import time
import os


def find_working_camera():
    """Find a working camera."""
    print("Searching for working camera...")
    
    for camera_id in range(5):  # Try cameras 0-4
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
    """Main function for bottle detection."""
    print("=" * 60)
    print("Bottle Detection with YOLO12")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO12 model
    print("Loading YOLO12 model...")
    model = YOLO('yolo12n.pt')
    
    # Find working camera
    camera_id = find_working_camera()
    
    # Setup camera
    print(f"Setting up camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully!")
    print("Controls:")
    print("  'q' - quit")
    print("  's' - save screenshot") 
    print("  'c' - switch camera")
    print("  '1-4' - select specific camera")
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0  # Initialize fps variable
    bottle_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            # Count bottles in current frame
            current_bottles = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        
                        # Check if it's a bottle (class 39 in COCO dataset)
                        if class_id == 39 and confidence > 0.5:  # Bottle class with confidence > 50%
                            current_bottles += 1
            
            # Update total bottle count
            if current_bottles > bottle_count:
                bottle_count = current_bottles
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add bottle counter
            cv2.putText(annotated_frame, f"Bottles detected: {current_bottles}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Max bottles: {bottle_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}, Bottles: {current_bottles}")
            
            # Display FPS and camera info on frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Camera: {camera_id}", (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display instructions
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Bottle Detection - YOLO12', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"bottle_detection_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                # Switch to next camera
                cap.release()
                camera_id = (camera_id + 1) % 5
                print(f"Switching to camera {camera_id}...")
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"Camera {camera_id} not available, switching back...")
                    camera_id = 0
                    cap = cv2.VideoCapture(camera_id)
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
            elif key >= ord('1') and key <= ord('4'):
                # Select specific camera
                new_camera_id = key - ord('0')
                if new_camera_id != camera_id:
                    cap.release()
                    camera_id = new_camera_id
                    print(f"Switching to camera {camera_id}...")
                    cap = cv2.VideoCapture(camera_id)
                    if not cap.isOpened():
                        print(f"Camera {camera_id} not available, switching back...")
                        camera_id = 0
                        cap = cv2.VideoCapture(camera_id)
                    else:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
    
    except KeyboardInterrupt:
        print("\\nDetection stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera released. Max bottles detected: {bottle_count}")


if __name__ == "__main__":
    main()
