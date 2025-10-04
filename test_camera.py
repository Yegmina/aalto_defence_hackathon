#!/usr/bin/env python3
"""
Test Camera with YOLO12

Simple script to test camera feed with YOLO12 detection.
"""

import cv2
from ultralytics import YOLO
import time


def main():
    """Test camera with YOLO12."""
    print("Testing Camera with YOLO12...")
    
    # Load YOLO12 model
    model = YOLO('yolo12n.pt')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Show frame
        cv2.imshow('YOLO12 Camera Test', annotated_frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed!")


if __name__ == "__main__":
    main()

