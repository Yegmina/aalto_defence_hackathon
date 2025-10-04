# Data Augmentation Module

This module handles automated data augmentation for YOLO object detection datasets. It uses the Albumentations library to apply various transformations while preserving YOLO annotation format.

## Files Overview

### Core Scripts
- **`augment_data.py`**: Main augmentation script with comprehensive transformations
- **`augment_new_dataset.py`**: Specialized augmentation for new datasets
- **`easy_start.py`**: Quick setup utility for beginners

### Utilities
- **`run_augmentation.bat`**: Windows batch file for easy execution

## Features

### Supported Transformations
- **Geometric**: Rotation, scaling, translation, flipping
- **Color**: Brightness, contrast, saturation, hue adjustments
- **Noise**: Gaussian noise, blur effects
- **Weather**: Rain, fog, snow simulation
- **Lighting**: Shadow, highlight adjustments

### Annotation Handling
- **YOLO Format**: Maintains class_id x_center y_center width height format
- **Bounding Box Preservation**: Ensures annotations remain valid after transformations
- **Multi-object Support**: Handles multiple objects per image
- **Validation**: Checks annotation validity after augmentation

## Usage

### Basic Augmentation
```bash
python augment_data.py --input_dir ./input --output_dir ./output --num_augmentations 5
```

### Parameters
- `--input_dir`: Directory containing input images and labels
- `--output_dir`: Directory for augmented output
- `--num_augmentations`: Number of augmented versions per image
- `--seed`: Random seed for reproducible results

### Quick Start
```bash
python easy_start.py
```
This script automatically organizes input data and creates placeholder labels if needed.

## Directory Structure

### Input Structure
```
input/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Output Structure
```
output/
├── images/
│   ├── image1_aug_1.jpg
│   ├── image1_aug_2.jpg
│   ├── image2_aug_1.jpg
│   └── ...
└── labels/
    ├── image1_aug_1.txt
    ├── image1_aug_2.txt
    ├── image2_aug_1.txt
    └── ...
```

## Augmentation Pipeline

### Transformation Sequence
1. **Input Validation**: Check image and label file compatibility
2. **Annotation Parsing**: Read YOLO format annotations
3. **Transformation Application**: Apply selected augmentations
4. **Annotation Update**: Update bounding box coordinates
5. **Output Saving**: Save augmented images and labels

### Quality Assurance
- **Bounding Box Validation**: Ensures boxes remain within image bounds
- **Annotation Format Check**: Validates YOLO format compliance
- **Image Quality**: Maintains image quality standards
- **Progress Tracking**: Real-time progress monitoring

## Configuration

### Default Settings
- **Rotation**: ±15 degrees
- **Scaling**: 0.8-1.2x
- **Brightness**: ±20%
- **Contrast**: ±20%
- **Noise**: Gaussian noise with variance 5-25

### Customization
Modify the augmentation pipeline in `augment_data.py`:
```python
transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    # Add more transformations as needed
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

## Performance

### Processing Speed
- **Images per second**: 3-5 images/second (depending on hardware)
- **Memory usage**: ~500MB for 1000 images
- **Output size**: 2-3x input size (depending on augmentation count)

### Optimization Tips
- Use SSD storage for faster I/O
- Increase batch size for better GPU utilization
- Use multiprocessing for CPU-bound operations
- Compress images if storage is limited

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size or number of augmentations
2. **Annotation errors**: Check label file format and encoding
3. **Slow processing**: Use GPU acceleration or reduce image resolution
4. **Invalid bounding boxes**: Adjust transformation parameters

### Error Messages
- **"Could not read annotation file"**: Check file format and encoding
- **"Invalid bounding box"**: Verify annotation coordinates
- **"Out of memory"**: Reduce batch size or image resolution

## Integration with Training

### YOLO Training
Augmented data can be directly used for YOLO training:
```bash
# After augmentation
python train_yolo12.py --data dataset.yaml
```

### Dataset Configuration
Update `dataset.yaml` to include augmented data:
```yaml
train: ./output/images
val: ./input/images
nc: 2
names: ['bottle', 'cup']
```

## Best Practices

### Data Quality
- Use diverse augmentation parameters
- Maintain annotation accuracy
- Validate augmented data before training
- Keep original data as backup

### Performance
- Use appropriate augmentation intensity
- Balance augmentation variety with processing time
- Monitor memory usage during processing
- Use reproducible seeds for consistent results

## Examples

### Basic Usage
```python
from augment_data import augment_dataset

# Augment dataset
augment_dataset(
    input_dir='./input',
    output_dir='./output',
    num_augmentations=5
)
```

### Advanced Usage
```python
import albumentations as A
from augment_data import create_augmentation_pipeline

# Create custom pipeline
transform = create_augmentation_pipeline(
    rotation_limit=30,
    scale_limit=0.3,
    brightness_limit=0.3
)

# Apply to single image
augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
```

## Dependencies

### Required Packages
- `albumentations`: Data augmentation library
- `opencv-python`: Image processing
- `numpy`: Numerical operations
- `Pillow`: Image handling
- `tqdm`: Progress bars

### Installation
```bash
pip install -r requirements.txt
```

## Support

For issues related to data augmentation:
1. Check the troubleshooting section
2. Verify input data format
3. Review augmentation parameters
4. Check system resources (memory, storage)
