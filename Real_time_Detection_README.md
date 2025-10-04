# Real-time Detection Module

This module provides comprehensive real-time object detection capabilities using trained YOLO12 models. It includes multiple detection scripts with varying complexity levels, from basic detection to advanced tracking and data export.

## Files Overview

### Advanced Detection Scripts
- **`realtime_new_detection.py`**: Full-featured detection with tracking and data export
- **`enhanced_bottle_detection.py`**: Enhanced detection with multi-class support
- **`quick_realtime_detection.py`**: Simple detection using pre-trained models

### Basic Detection Scripts
- **`bottle_detection.py`**: Basic bottle detection with camera switching
- **`simple_bottle_detection.py`**: Minimal detection implementation
- **`live_detection.py`**: Live camera feed processing

### Testing and Utilities
- **`test_camera.py`**: Camera functionality testing
- **`test_bottle_detection.py`**: Detection system testing

## Features

### Core Detection Capabilities
- **Real-time Processing**: 25-30 FPS on standard hardware
- **Object Tracking**: Persistent object identification across frames
- **Multi-class Detection**: Support for multiple object classes
- **Confidence Filtering**: Configurable detection thresholds
- **Bounding Box Visualization**: Real-time annotation overlay

### Camera Management
- **Auto-detection**: Automatic working camera discovery
- **Multi-camera Support**: Switch between available cameras
- **Camera Switching**: Runtime camera selection (keys 1-4)
- **Fallback Handling**: Graceful handling of camera failures

### Data Export and Analysis
- **Detection Masks**: Save annotated frames
- **JSON Metadata**: Export detection results and statistics
- **Performance Metrics**: FPS, detection counts, and timing
- **Sequence Tracking**: Persistent object tracking across sessions

## Usage

### Advanced Detection
```bash
python realtime_new_detection.py
```

### Quick Detection
```bash
python quick_realtime_detection.py
```

### Basic Detection
```bash
python bottle_detection.py
```

## Controls

### Keyboard Controls
- **'q'**: Quit detection
- **'c'**: Cycle through available cameras
- **'1'-'4'**: Select camera directly
- **'r'**: Reset detection sequence
- **'s'**: Save current frame

### Mouse Controls
- **Click**: Select detection area (if supported)
- **Scroll**: Adjust detection threshold (if supported)

## Configuration

### Detection Parameters
```python
# Confidence threshold (0.0-1.0)
CONF_THRESHOLD = 0.5

# IoU threshold for NMS (0.0-1.0)
IOU_THRESHOLD = 0.45

# Allowed detection classes
ALLOWED_LABELS = ['cup', 'bottle']

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

### Model Configuration
```python
# Model loading priority
MODEL_PATHS = [
    'runs/train/yolo12_bottle_cup_new/weights/best.pt',  # Fine-tuned model
    'yolo12s.pt',  # YOLO12 small
    'yolo12n.pt'   # YOLO12 nano (fallback)
]
```

## Performance Optimization

### Hardware Optimization
- **GPU Acceleration**: Use CUDA for faster inference
- **CPU Optimization**: Use multiple cores for processing
- **Memory Management**: Optimize buffer sizes
- **Storage**: Use SSD for faster data export

### Software Optimization
- **Model Size**: Use smaller models for faster inference
- **Image Resolution**: Reduce resolution for higher FPS
- **Batch Processing**: Process multiple frames simultaneously
- **Threading**: Use separate threads for I/O and processing

### Performance Targets
- **FPS**: 25-30 FPS (laptop), 15-20 FPS (Raspberry Pi)
- **Latency**: <50ms end-to-end processing
- **Memory**: <2GB RAM usage
- **CPU**: <50% CPU utilization

## Detection Pipeline

### Frame Processing
1. **Frame Capture**: Read frame from camera
2. **Preprocessing**: Resize and normalize image
3. **Inference**: Run YOLO model prediction
4. **Post-processing**: Apply NMS and filtering
5. **Tracking**: Update object tracks
6. **Visualization**: Draw bounding boxes and labels
7. **Export**: Save detection data (if enabled)

### Object Tracking
```python
# Enable tracking
results = model.track(
    frame,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    persist=True,
    verbose=False
)
```

### Data Export
```python
# Save detection mask
cv2.imwrite(f'detection_{timestamp}.jpg', annotated_frame)

# Save JSON metadata
metadata = {
    'timestamp': timestamp,
    'camera_id': camera_id,
    'fps': fps,
    'detections': detection_results
}
```

## Camera Management

### Auto-detection
```python
def find_working_camera():
    for camera_id in range(4):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return camera_id
        cap.release()
    return 0  # Default fallback
```

### Camera Switching
```python
def switch_camera(camera_id):
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
    return None
```

## Statistics and Monitoring

### Real-time Statistics
- **Detection Count**: Number of objects detected
- **FPS**: Current frames per second
- **Camera ID**: Active camera identifier
- **Sequence Number**: Detection session counter
- **Class-specific Counts**: Per-class detection statistics

### Performance Metrics
- **Processing Time**: Time per frame
- **Memory Usage**: RAM and VRAM utilization
- **CPU Usage**: Processor utilization
- **Detection Accuracy**: Confidence scores and counts

## Data Export

### Export Formats
- **Images**: Annotated detection frames (JPG)
- **JSON**: Detection metadata and statistics
- **CSV**: Tabular detection data (if supported)
- **Video**: Recorded detection sessions (if supported)

### Export Structure
```
detection_output/
├── detection_20250104_120000_001.jpg
├── detection_20250104_120000_001.json
├── detection_20250104_120001_002.jpg
├── detection_20250104_120001_002.json
└── ...
```

### JSON Metadata Format
```json
{
  "timestamp": "20250104_120000_001",
  "camera_id": 0,
  "fps": 25.3,
  "sequence": 1,
  "detections": [
    {
      "class_id": 0,
      "class_name": "bottle",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 300],
      "track_id": 1
    }
  ]
}
```

## Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera permissions and drivers
2. **Low FPS**: Reduce image resolution or use GPU acceleration
3. **Memory errors**: Reduce buffer sizes or use smaller model
4. **Detection failures**: Check model path and confidence thresholds

### Performance Issues
1. **High CPU usage**: Enable GPU acceleration or reduce processing
2. **Memory leaks**: Check for proper resource cleanup
3. **Slow startup**: Preload model or use model caching
4. **Inconsistent FPS**: Optimize processing pipeline

### Error Messages
- **"Could not open camera"**: Check camera availability and permissions
- **"Model not found"**: Verify model path and file existence
- **"Out of memory"**: Reduce image resolution or batch size
- **"Invalid frame"**: Check camera connection and settings

## Integration with Training

### Model Loading
```python
# Load fine-tuned model
model = YOLO('runs/train/yolo12_bottle_cup_new/weights/best.pt')

# Load with fallback
model_paths = [
    'runs/train/yolo12_bottle_cup_new/weights/best.pt',
    'yolo12s.pt',
    'yolo12n.pt'
]
```

### Model Validation
```python
# Test model before deployment
results = model('test_image.jpg')
print(f"Detections: {len(results[0].boxes)}")
```

## Raspberry Pi Deployment

### Pi-optimized Detection
```bash
python pi_bottle_detection.py
```

### Performance Considerations
- **Model Size**: Use YOLO12n (nano) for Pi deployment
- **Resolution**: Reduce to 320x240 for better performance
- **FPS**: Expect 10-15 FPS on Pi 4
- **Memory**: Monitor RAM usage (4GB Pi recommended)

## Best Practices

### Detection Quality
- Use appropriate confidence thresholds
- Implement proper NMS settings
- Validate detection results
- Monitor false positive rates

### Performance
- Optimize for target hardware
- Use appropriate model size
- Balance accuracy and speed
- Monitor resource usage

### User Experience
- Provide clear visual feedback
- Implement intuitive controls
- Handle errors gracefully
- Provide performance metrics

## Dependencies

### Required Packages
- `ultralytics`: YOLO detection framework
- `opencv-python`: Camera and image processing
- `numpy`: Numerical operations
- `torch`: PyTorch framework
- `Pillow`: Image handling
- `json`: Data serialization
- `datetime`: Timestamp handling
- `collections`: Data structures
- `time`: Timing utilities

### Installation
```bash
pip install -r requirements.txt
pip install -r requirements_yolo.txt
```

## Support

For detection-related issues:
1. Check camera connectivity and permissions
2. Verify model file existence and format
3. Review detection parameters and thresholds
4. Monitor system resources and performance
5. Check error logs for specific issues
