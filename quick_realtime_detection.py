#!/usr/bin/env python3
"""
Quick Real-time Detection with Pre-trained YOLO12 Model

This script performs real-time object detection using the pre-trained YOLO12 model
from test3/yolo12n_finetune/weights/best.pt
"""

import cv2
import torch
from ultralytics import YOLO
import time
import os


def find_working_camera():
    """Find a working camera."""
    print("Searching for working camera...")
    
    for camera_id in range(4):
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Found working camera at index {camera_id}")
                    cap.release()
                    return camera_id
                cap.release()
        except Exception as e:
            print(f"Camera {camera_id} failed: {e}")
            continue
    
    print("No working camera found, using camera 0 as default")
    return 0


def main():
    """Main function for quick real-time detection."""
    print("=" * 60)
    print("Quick Real-time Detection with Pre-trained YOLO12")
    print("=" * 60)
    
    # Model path
    model_path = "test3/yolo12n_finetune/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please make sure the model file exists.")
        return
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Find working camera
    camera_id = find_working_camera()
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Camera {camera_id} initialized")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'c' - Switch camera")
    print("  's' - Save current frame")
    print("=" * 60)
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Run inference
            results = model.track(
                frame,
                conf=0.5,  # Confidence threshold
                iou=0.45,  # IoU threshold
                persist=True,
                verbose=False
            )
            
            # Draw detections
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                class_names = model.names
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by confidence
                    if confidence < 0.5:
                        continue
                    
                    # Get class name
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # Draw bounding box
                    color = (0, 255, 0) if class_name == "cup" else (255, 0, 0)  # Green for cup, Blue for bottle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label
                    label = f"{class_name} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                 (int(x1) + label_size[0], int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = current_time
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Camera: {camera_id}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Quick Real-time Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Switch camera
                cap.release()
                camera_id = (camera_id + 1) % 4
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"Could not open camera {camera_id}")
                    break
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Switched to camera {camera_id}")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame: {filename}")
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Detection session ended")


if __name__ == "__main__":
    main()
