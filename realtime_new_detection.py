#!/usr/bin/env python3
"""
Real-time Object Detection with Fine-tuned YOLO12 Model

This script performs real-time object detection using the fine-tuned YOLO12 model
trained on the new bottle/cup dataset. It includes object tracking, configurable
thresholds, and data export capabilities.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import json
import os
from datetime import datetime
import time
from collections import defaultdict, deque


class RealTimeDetector:
    def __init__(self, model_path=None, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the real-time detector.
        
        Args:
            model_path: Path to the fine-tuned model (optional)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.camera = None
        self.camera_id = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = deque(maxlen=30)
        
        # Detection statistics
        self.detection_stats = defaultdict(int)
        self.max_detections = defaultdict(int)
        self.total_detections = 0
        self.sequence_count = 0
        
        # Data export
        self.export_dir = "detection_output"
        self.setup_export_directory()
        
        # Load model
        self.load_model(model_path)
        
        # Camera management
        self.find_working_camera()
    
    def setup_export_directory(self):
        """Create export directory if it doesn't exist."""
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
            print(f"Created export directory: {self.export_dir}")
    
    def load_model(self, model_path=None):
        """Load the YOLO model with fallback options."""
        model_paths = [
            model_path,
            "runs/train/yolo12_bottle_cup_new/weights/best.pt",
            "yolo12s.pt",
            "yolo12n.pt"
        ]
        
        for path in model_paths:
            if path and os.path.exists(path):
                try:
                    print(f"Loading model: {path}")
                    self.model = YOLO(path)
                    print(f"Successfully loaded model: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        # Fallback to default model
        print("Loading default YOLO12n model...")
        self.model = YOLO("yolo12n.pt")
        print("Loaded default YOLO12n model")
    
    def find_working_camera(self):
        """Find a working camera."""
        print("Searching for working camera...")
        
        for camera_id in range(4):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Found working camera at index {camera_id}")
                        self.camera_id = camera_id
                        cap.release()
                        return
                cap.release()
            except Exception as e:
                print(f"Camera {camera_id} failed: {e}")
                continue
        
        print("No working camera found, using camera 0 as default")
        self.camera_id = 0
    
    def switch_camera(self, camera_id):
        """Switch to a different camera."""
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    if self.camera:
                        self.camera.release()
                    self.camera_id = camera_id
                    self.camera = cap
                    print(f"Switched to camera {camera_id}")
                    return True
                cap.release()
        except Exception as e:
            print(f"Failed to switch to camera {camera_id}: {e}")
        
        return False
    
    def cycle_camera(self):
        """Cycle to the next available camera."""
        for camera_id in range(4):
            next_id = (self.camera_id + camera_id + 1) % 4
            if self.switch_camera(next_id):
                return True
        return False
    
    def calculate_fps(self):
        """Calculate and update FPS."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.fps_history.append(self.fps)
            self.frame_count = 0
            self.start_time = current_time
        
        # Use average FPS from history if available
        if self.fps_history:
            self.fps = sum(self.fps_history) / len(self.fps_history)
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on the frame."""
        if not results or not results[0].boxes:
            return frame
        
        boxes = results[0].boxes
        class_names = self.model.names
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Filter by confidence threshold
            if confidence < self.conf_threshold:
                continue
            
            # Get class name
            class_name = class_names.get(class_id, f"class_{class_id}")
            
            # Update statistics
            self.detection_stats[class_name] += 1
            self.total_detections += 1
            
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
        
        return frame
    
    def draw_statistics(self, frame):
        """Draw detection statistics on the frame."""
        # Update max detections
        for class_name, count in self.detection_stats.items():
            self.max_detections[class_name] = max(self.max_detections[class_name], count)
        
        # Draw statistics
        y_offset = 30
        stats_text = [
            f"Total: {self.total_detections}",
            f"Sequence: {self.sequence_count}",
            f"Camera: {self.camera_id}",
            f"FPS: {self.fps:.1f}"
        ]
        
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
        
        # Draw class-specific statistics
        y_offset += 10
        for class_name in ["cup", "bottle"]:
            current = self.detection_stats[class_name]
            max_count = self.max_detections[class_name]
            text = f"{class_name.capitalize()}s: {current}"
            if max_count > 0:
                text += f" (Max: {max_count})"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def export_detection_data(self, frame, results):
        """Export detection data to files."""
        if not results or not results[0].boxes:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Save detection mask
        mask_path = os.path.join(self.export_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(mask_path, frame)
        
        # Save JSON metadata
        json_path = os.path.join(self.export_dir, f"detection_{timestamp}.json")
        metadata = {
            "timestamp": timestamp,
            "camera_id": self.camera_id,
            "fps": self.fps,
            "sequence": self.sequence_count,
            "detections": []
        }
        
        boxes = results[0].boxes
        for box in boxes:
            if box.conf[0].cpu().numpy() >= self.conf_threshold:
                detection = {
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "class_name": self.model.names.get(int(box.cls[0].cpu().numpy()), "unknown"),
                    "confidence": float(box.conf[0].cpu().numpy()),
                    "bbox": box.xyxy[0].cpu().numpy().tolist()
                }
                metadata["detections"].append(detection)
        
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def reset_sequence(self):
        """Reset detection sequence and statistics."""
        self.sequence_count += 1
        self.detection_stats.clear()
        self.total_detections = 0
        print(f"Reset sequence to {self.sequence_count}")
    
    def run(self):
        """Run the real-time detection loop."""
        print("Starting real-time detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Cycle camera")
        print("  '1'-'4' - Select camera directly")
        print("  'r' - Reset sequence")
        print("  's' - Save current frame")
        
        # Initialize camera
        self.camera = cv2.VideoCapture(self.camera_id)
        
        if not self.camera.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Camera {self.camera_id} initialized")
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Run inference
                results = self.model.track(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    verbose=False
                )
                
                # Draw detections
                frame = self.draw_detections(frame, results)
                
                # Draw statistics
                frame = self.draw_statistics(frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Export data (every 10th frame to avoid too many files)
                if self.frame_count % 10 == 0:
                    self.export_detection_data(frame, results)
                
                # Display frame
                cv2.imshow("Real-time Detection", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.cycle_camera()
                elif key == ord('r'):
                    self.reset_sequence()
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
                elif key >= ord('1') and key <= ord('4'):
                    camera_id = key - ord('1')
                    self.switch_camera(camera_id)
        
        except KeyboardInterrupt:
            print("\nDetection stopped by user")
        
        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Detection session ended")


def main():
    """Main function."""
    print("=" * 60)
    print("Real-time Object Detection with Fine-tuned YOLO12")
    print("=" * 60)
    
    # Configuration
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    print(f"Configuration:")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print(f"  Model: Fine-tuned YOLO12 (bottle/cup detection)")
    print("=" * 60)
    
    # Create and run detector
    detector = RealTimeDetector(
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    detector.run()


if __name__ == "__main__":
    main()