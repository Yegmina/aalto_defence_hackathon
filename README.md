# YOLO12 Object Detection Pipeline - Defence Hackathon

A comprehensive object detection pipeline using YOLO12 for real-time bottle and cup detection, developed during the Defence Hackathon. This project includes data augmentation, model training, and real-time detection capabilities.

## Project Overview

This project was developed to create a robust object detection system capable of distinguishing between bottles and cups in real-time. The pipeline includes:

- **Data Augmentation**: Automated image and annotation augmentation using Albumentations
- **Model Training**: YOLO12 fine-tuning on custom datasets
- **Real-time Detection**: Live camera feed processing with object tracking
- **Raspberry Pi Support**: Optimized deployment for edge devices

## Repository Structure

```
defence_hackathon/
├── README.md                           # This file
├── requirements.txt                    # Basic dependencies
├── requirements_yolo.txt              # YOLO training dependencies
├── dataset.yaml                       # Original dataset configuration
├── new_dataset.yaml                   # New dataset configuration
│
├── Data Augmentation/
│   ├── augment_data.py                # Main augmentation script
│   ├── augment_new_dataset.py         # New dataset augmentation
│   ├── easy_start.py                  # Quick setup utility
│   └── run_augmentation.bat           # Windows batch file
│
├── Model Training/
│   ├── setup_yolo12.py               # YOLO12 environment setup
│   ├── train_yolo12.py               # Main training script
│   ├── train_new_yolo12.py           # New dataset training
│   ├── quick_start_yolo.py           # Quick training start
│   ├── check_setup.py                # Environment verification
│   └── start_yolo_training.bat       # Windows batch file
│
├── Real-time Detection/
│   ├── realtime_new_detection.py     # Advanced detection with tracking
│   ├── quick_realtime_detection.py   # Simple detection script
│   ├── enhanced_bottle_detection.py  # Enhanced features
│   ├── bottle_detection.py           # Basic bottle detection
│   ├── simple_bottle_detection.py    # Minimal detection
│   ├── live_detection.py             # Live camera detection
│   └── test_camera.py                # Camera testing utility
│
├── Utilities/
│   ├── fix_dataset_labels.py         # Label format fixing
│   ├── fix_corrupted_labels.py       # Corrupted label repair
│   ├── update_bottle_labels.py       # Label updating utility
│   ├── setup_new_dataset.py          # New dataset preparation
│   └── test_bottle_detection.py      # Detection testing
│
├── Raspberry Pi Deployment/
│   ├── complete_pi_package/          # Complete Pi deployment package
│   ├── pi_bottle_detection.py        # Pi-optimized detection
│   ├── transfer_to_pi.sh             # Transfer script
│   └── setup_raspberry_pi.py         # Pi setup automation
│
└── GitHub Management/
    ├── push_to_github.py             # Code push utility
    ├── simple_github_push.py         # Simplified push script
    └── remove_images_from_github.py  # Image cleanup utility
```

## Quick Start

### 1. Environment Setup

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install YOLO training dependencies
pip install -r requirements_yolo.txt
```

### 2. Data Augmentation

```bash
# Run augmentation on input data
python augment_data.py --input_dir ./input --output_dir ./output --num_augmentations 5

# Or use the quick start
python easy_start.py
```

### 3. Model Training

```bash
# Setup YOLO12 environment
python setup_yolo12.py

# Start training
python train_yolo12.py

# Or use quick start
python quick_start_yolo.py
```

### 4. Real-time Detection

```bash
# Run advanced detection with tracking
python realtime_new_detection.py

# Or use simple detection
python quick_realtime_detection.py
```

## Hackathon Development Process

### Phase 1: Initial Setup and Data Preparation
- Set up YOLO12 environment and dependencies
- Organized input data with proper directory structure
- Created data augmentation pipeline using Albumentations
- Implemented YOLO annotation format handling

### Phase 2: Model Training and Optimization
- Trained initial YOLO12 model on custom bottle dataset
- Implemented dataset splitting (train/val/test)
- Created training monitoring and validation scripts
- Optimized training parameters for better performance

### Phase 3: Real-time Detection Development
- Developed multiple detection scripts with varying complexity
- Implemented object tracking for persistent detection
- Added camera management and switching capabilities
- Created data export functionality for analysis

### Phase 4: Label Correction and Model Improvement
- Identified and fixed label format issues
- Corrected class label mismatches (bottle/cup confusion)
- Retrained model with corrected annotations
- Achieved 94.72% mAP50 accuracy

### Phase 5: Raspberry Pi Deployment
- Created Pi-optimized detection scripts
- Developed automated setup and transfer utilities
- Implemented lightweight detection for edge devices
- Created comprehensive deployment documentation

### Phase 6: Code Organization and Documentation
- Organized code into logical modules
- Created comprehensive documentation
- Implemented GitHub management utilities
- Cleaned repository of large data files

## Key Features

### Data Augmentation
- **Comprehensive Transformations**: Rotation, scaling, brightness, contrast, noise
- **Annotation Preservation**: Maintains YOLO format throughout augmentation
- **Batch Processing**: Efficient processing of large datasets
- **Progress Tracking**: Real-time progress bars and statistics

### Model Training
- **YOLO12 Integration**: Latest YOLO architecture with improved performance
- **Custom Dataset Support**: Flexible dataset configuration
- **Training Monitoring**: Real-time loss and accuracy tracking
- **Model Validation**: Comprehensive validation metrics

### Real-time Detection
- **Object Tracking**: Persistent object identification across frames
- **Multi-camera Support**: Automatic camera detection and switching
- **Configurable Thresholds**: Adjustable confidence and IoU thresholds
- **Data Export**: Detection results and metadata saving
- **Performance Metrics**: FPS monitoring and statistics

### Raspberry Pi Deployment
- **Optimized Performance**: Lightweight detection for edge devices
- **Automated Setup**: One-click installation and configuration
- **Network Configuration**: WiFi and SSH setup automation
- **Transfer Utilities**: Easy code deployment to Pi devices

## Performance Results

### Model Performance
- **mAP50**: 94.72% (excellent accuracy)
- **mAP50-95**: 62.76% (good overall performance)
- **Precision**: 83.36% (low false positives)
- **Recall**: 92.54% (high detection rate)

### Real-time Performance
- **FPS**: 25-30 FPS on laptop hardware
- **Latency**: <50ms inference time
- **Memory Usage**: Optimized for 4GB+ RAM systems
- **Camera Support**: Multi-camera detection and switching

## Technical Specifications

### Hardware Requirements
- **Minimum**: 4GB RAM, CPU with AVX support
- **Recommended**: 8GB+ RAM, NVIDIA GPU with CUDA support
- **Raspberry Pi**: Pi 4 Model B with 4GB+ RAM

### Software Dependencies
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Ultralytics**: Latest version
- **OpenCV**: 4.5+
- **Albumentations**: Latest version

### Supported Formats
- **Images**: JPG, PNG, BMP, TIFF
- **Annotations**: YOLO format (.txt)
- **Models**: PyTorch (.pt), ONNX (.onnx)
- **Configurations**: YAML (.yaml)

## Usage Examples

### Basic Detection
```python
from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Run detection
results = model('image.jpg')
results.show()
```

### Real-time Detection
```python
# Run real-time detection
python realtime_new_detection.py

# Controls:
# 'q' - Quit
# 'c' - Switch camera
# 'r' - Reset sequence
# 's' - Save frame
```

### Data Augmentation
```python
# Run augmentation
python augment_data.py --input_dir ./input --output_dir ./output --num_augmentations 5

# Parameters:
# --input_dir: Input data directory
# --output_dir: Output directory
# --num_augmentations: Number of augmented versions per image
```

## Troubleshooting

### Common Issues
1. **Camera not detected**: Check camera permissions and drivers
2. **Low FPS**: Reduce image resolution or use GPU acceleration
3. **Memory errors**: Reduce batch size or use smaller model
4. **Label format errors**: Run fix_corrupted_labels.py

### Performance Optimization
- Use GPU acceleration for training and inference
- Reduce image resolution for faster processing
- Use smaller YOLO model variants (nano, small)
- Optimize batch sizes for available memory

## Contributing

This project was developed during the Defence Hackathon. For contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is developed for educational and research purposes during the Defence Hackathon.

## Acknowledgments

- **Ultralytics**: YOLO12 implementation and training framework
- **Albumentations**: Data augmentation library
- **OpenCV**: Computer vision and camera handling
- **PyTorch**: Deep learning framework
- **Defence Hackathon**: Event organization and support

## Contact

For questions or support regarding this project, please refer to the hackathon documentation or contact the development team.