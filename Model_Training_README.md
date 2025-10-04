# Model Training Module

This module handles YOLO12 model training, fine-tuning, and optimization for custom object detection datasets. It provides comprehensive training pipelines with monitoring and validation capabilities.

## Files Overview

### Core Training Scripts
- **`train_yolo12.py`**: Main training script with full configuration
- **`train_new_yolo12.py`**: Specialized training for new datasets
- **`quick_start_yolo.py`**: Simplified training with optimized settings

### Setup and Configuration
- **`setup_yolo12.py`**: Environment setup and dataset preparation
- **`check_setup.py`**: Environment verification and diagnostics
- **`start_yolo_training.bat`**: Windows batch file for easy execution

## Features

### Training Capabilities
- **YOLO12 Integration**: Latest YOLO architecture with improved performance
- **Custom Dataset Support**: Flexible dataset configuration
- **Transfer Learning**: Pre-trained model fine-tuning
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Mixed Precision**: Automatic mixed precision for faster training

### Monitoring and Validation
- **Real-time Metrics**: Loss, accuracy, and performance tracking
- **Validation Monitoring**: Automatic validation during training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Model Checkpointing**: Automatic best model saving
- **Training Visualization**: Loss curves and metrics plots

## Usage

### Basic Training
```bash
python train_yolo12.py
```

### Quick Start Training
```bash
python quick_start_yolo.py
```

### New Dataset Training
```bash
python train_new_yolo12.py
```

## Configuration

### Dataset Configuration (dataset.yaml)
```yaml
# Dataset configuration
path: ./yolo_dataset
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 2

# Class names
names:
  0: bottle
  1: cup
```

### Training Parameters
- **Epochs**: 100 (default)
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Image Size**: 640x640 (YOLO standard)
- **Learning Rate**: 0.01 (with cosine annealing)
- **Optimizer**: AdamW with weight decay
- **Augmentation**: Built-in YOLO augmentations

### Advanced Configuration
```python
# Custom training parameters
model.train(
    data='dataset.yaml',
    epochs=100,
    batch=16,
    imgsz=640,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.0,
    nbs=64,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,
    val=True
)
```

## Training Process

### Phase 1: Data Preparation
1. **Dataset Validation**: Check image and label file compatibility
2. **Data Splitting**: Automatic train/val/test split
3. **Annotation Verification**: Validate YOLO format annotations
4. **Cache Creation**: Build dataset cache for faster loading

### Phase 2: Model Initialization
1. **Model Loading**: Load pre-trained YOLO12 weights
2. **Architecture Adaptation**: Adjust for custom number of classes
3. **Transfer Learning**: Initialize with COCO pre-trained weights
4. **Optimizer Setup**: Configure AdamW optimizer with scheduling

### Phase 3: Training Loop
1. **Forward Pass**: Model inference on training batch
2. **Loss Calculation**: Compute detection loss (box, class, DFL)
3. **Backward Pass**: Gradient computation and optimization
4. **Validation**: Periodic validation on held-out data
5. **Checkpointing**: Save best model based on validation metrics

### Phase 4: Post-Training
1. **Model Validation**: Final validation on test set
2. **Metrics Calculation**: Compute mAP, precision, recall
3. **Model Export**: Save final model in various formats
4. **Results Visualization**: Generate training plots and reports

## Performance Metrics

### Training Metrics
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss
- **DFL Loss**: Distribution Focal Loss
- **Total Loss**: Combined loss for optimization

### Validation Metrics
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Target Performance
- **mAP50**: >90% (excellent)
- **mAP50-95**: >60% (good)
- **Precision**: >80% (low false positives)
- **Recall**: >85% (high detection rate)

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB system memory
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with 4GB VRAM (optional but recommended)
- **CPU**: Multi-core processor with AVX support

### Recommended Configuration
- **RAM**: 16GB+ system memory
- **Storage**: SSD with 50GB+ free space
- **GPU**: NVIDIA RTX 3060 or better
- **CPU**: Intel i7 or AMD Ryzen 7

### Performance Expectations
- **Training Time**: 2-4 hours for 100 epochs (GPU)
- **Memory Usage**: 4-8GB VRAM (depending on batch size)
- **Storage Usage**: 2-5GB for model and logs

## Optimization Strategies

### Training Optimization
- **Batch Size**: Increase for better GPU utilization
- **Mixed Precision**: Use AMP for faster training
- **Data Loading**: Use multiple workers for I/O
- **Gradient Accumulation**: Simulate larger batch sizes

### Memory Optimization
- **Image Size**: Reduce for lower memory usage
- **Batch Size**: Decrease if out of memory
- **Model Size**: Use smaller YOLO variants (nano, small)
- **Gradient Checkpointing**: Trade compute for memory

### Speed Optimization
- **GPU Acceleration**: Use CUDA for faster training
- **Data Prefetching**: Load next batch while training
- **Model Compilation**: Use torch.compile for faster inference
- **Distributed Training**: Use multiple GPUs if available

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Enable GPU acceleration or reduce model size
3. **Poor Performance**: Check dataset quality and annotations
4. **Training Instability**: Reduce learning rate or adjust scheduler

### Error Messages
- **"CUDA out of memory"**: Reduce batch size or use gradient accumulation
- **"No valid images found"**: Check dataset path and annotation format
- **"Invalid annotation"**: Verify YOLO format compliance
- **"Model not found"**: Check model path and file existence

## Model Evaluation

### Validation Process
```python
# Run validation
results = model.val(data='dataset.yaml')

# Print metrics
print(f"mAP50: {results.box.map50}")
print(f"mAP50-95: {results.box.map}")
print(f"Precision: {results.box.mp}")
print(f"Recall: {results.box.mr}")
```

### Model Testing
```python
# Test on single image
results = model('test_image.jpg')

# Test on dataset
results = model.val(data='test_dataset.yaml')
```

## Model Export

### Supported Formats
- **PyTorch**: .pt (default)
- **ONNX**: .onnx (for inference optimization)
- **TensorRT**: .engine (for NVIDIA GPUs)
- **TorchScript**: .torchscript (for deployment)

### Export Commands
```python
# Export to ONNX
model.export(format='onnx')

# Export to TensorRT
model.export(format='engine')

# Export to TorchScript
model.export(format='torchscript')
```

## Integration with Detection

### Model Loading
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/train/yolo12_bottle_cup/weights/best.pt')

# Run inference
results = model('image.jpg')
```

### Real-time Detection
```python
# Use in real-time detection
python realtime_new_detection.py --model runs/train/yolo12_bottle_cup/weights/best.pt
```

## Best Practices

### Dataset Preparation
- Use diverse, high-quality images
- Ensure accurate annotations
- Balance class distribution
- Include edge cases and difficult examples

### Training Strategy
- Start with pre-trained weights
- Use appropriate learning rates
- Monitor validation metrics
- Implement early stopping
- Save multiple checkpoints

### Model Selection
- Choose appropriate model size for hardware
- Consider inference speed requirements
- Balance accuracy and efficiency
- Test on real-world data

## Dependencies

### Required Packages
- `ultralytics`: YOLO training framework
- `torch`: PyTorch deep learning framework
- `torchvision`: Computer vision utilities
- `opencv-python`: Image processing
- `Pillow`: Image handling
- `numpy`: Numerical operations
- `PyYAML`: Configuration file handling
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualization
- `pandas`: Data manipulation
- `tqdm`: Progress bars

### Installation
```bash
pip install -r requirements_yolo.txt
```

## Support

For training-related issues:
1. Check hardware requirements
2. Verify dataset format and quality
3. Review training parameters
4. Monitor system resources
5. Check error logs for specific issues
