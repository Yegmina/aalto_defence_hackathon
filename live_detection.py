#!/usr/bin/env python3
"""
Live Object Detection with Camera

This script performs real-time object detection using YOLO12 and your webcam.
It will detect your target objects in the camera feed.
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import time


def setup_camera():
    """Setup camera capture."""
    cap = cv2.VideoCapture(0)  # Use default camera (usually webcam)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    return cap


def draw_detections(frame, results, class_names):
    """Draw bounding boxes and labels on the frame."""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_names[class_id]}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame


def main():
    """Main function for live detection."""
    print("=" * 60)
    print("Live Object Detection with YOLO12")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the trained model
    model_path = "runs/train/yolo12_quick/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Trained model not found at: {model_path}")
        print("Using pretrained YOLO12 model instead...")
        model_path = "yolo12n.pt"
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Setup camera
    print("Setting up camera...")
    cap = setup_camera()
    if cap is None:
        return
    
    # Class names
    class_names = {0: 'target_object'}  # Your class
    
    print("Starting live detection...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run inference
            results = model(frame, verbose=False)
            
            # Draw detections
            frame = draw_detections(frame, results, class_names)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('YOLO12 Live Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\\nDetection stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


if __name__ == "__main__":
    import os
    main()

