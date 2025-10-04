#!/usr/bin/env python3
"""
Simple Bottle Detection

A simplified version of bottle detection for easy testing.
"""

import cv2
from ultralytics import YOLO
import time


def main():
    """Simple bottle detection."""
    print("Starting Simple Bottle Detection...")
    
    # Load YOLO12 model
    model = YOLO('yolo12n.pt')
    
    # Open camera
    cap = cv2.VideoCapture(1)  # Use camera 1
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Show frame
        cv2.imshow('Bottle Detection', annotated_frame)
        
        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped!")


if __name__ == "__main__":
    main()

