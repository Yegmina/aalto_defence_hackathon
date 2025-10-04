#!/usr/bin/env python3
"""
Train YOLO12 on New Dataset

This script trains a YOLO12 model on the new bottle/cup dataset with augmented data.
"""

import torch
from ultralytics import YOLO
import os
import time


def main():
    """Main training function."""
    print("=" * 60)
    print("YOLO12 Training on New Bottle/Cup Dataset")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load YOLO12 model
    print("Loading YOLO12 model...")
    model = YOLO('yolo12n.pt')  # Start with nano for faster training
    
    # Check dataset configuration
    dataset_yaml = 'new_dataset.yaml'
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset configuration file '{dataset_yaml}' not found!")
        print("Please run setup_new_dataset.py first.")
        return
    
    print(f"Using dataset configuration: {dataset_yaml}")
    
    # Train the model
    print("Starting training...")
    start_time = time.time()
    
    results = model.train(
        data=dataset_yaml,                    # Dataset configuration
        epochs=100,                          # Number of training epochs
        imgsz=640,                          # Input image size
        batch=16,                           # Batch size (adjust based on GPU memory)
        device=device,                      # Device to use
        workers=4,                          # Number of worker threads
        project='runs/train',               # Project directory
        name='yolo12_bottle_cup_new',       # Experiment name
        exist_ok=True,                      # Overwrite existing experiment
        pretrained=True,                    # Use pretrained weights
        optimizer='AdamW',                  # Optimizer
        lr0=0.01,                          # Initial learning rate
        lrf=0.01,                          # Final learning rate (lr0 * lrf)
        momentum=0.937,                     # SGD momentum
        weight_decay=0.0005,                # Optimizer weight decay
        warmup_epochs=3,                    # Warmup epochs
        warmup_momentum=0.8,                # Warmup initial momentum
        warmup_bias_lr=0.1,                 # Warmup initial bias lr
        box=7.5,                           # Box loss gain
        cls=0.5,                           # Cls loss gain
        dfl=1.5,                           # Dfl loss gain
        pose=12.0,                         # Pose loss gain
        kobj=1.0,                          # Keypoint obj loss gain
        label_smoothing=0.0,                # Label smoothing
        nbs=64,                            # Nominal batch size
        overlap_mask=True,                 # Masks should overlap during training
        mask_ratio=4,                      # Mask downsample ratio
        dropout=0.0,                       # Use dropout regularization
        val=True,                          # Validate/test during training
        split='val',                       # Dataset split to use for validation
        save_json=False,                   # Save results to JSON
        save_hybrid=False,                 # Save hybrid version of labels
        conf=None,                         # Object confidence threshold
        iou=0.7,                           # NMS IoU threshold
        max_det=300,                       # Maximum detections per image
        half=False,                        # Use half precision (FP16)
        dnn=False,                         # Use OpenCV DNN for ONNX inference
        plots=True,                        # Save plots during training
        source=None,                       # Source directory for images or videos
        vid_stride=1,                      # Video frame-rate stride
        stream_buffer=False,               # Buffer all streaming images
        visualize=False,                   # Visualize model features
        augment=False,                     # Apply image augmentation to prediction sources
        agnostic_nms=False,                # Class-agnostic NMS
        retina_masks=False,                # Use high-resolution segmentation masks
        embed=None,                        # Return feature vectors/embeddings from given layers
        show=False,                        # Show results if possible
        save_frames=False,                 # Save video frames
        save_txt=False,                    # Save results as .txt file
        save_conf=False,                   # Save results with confidence scores
        save_crop=False,                   # Save cropped images with results
        show_labels=True,                  # Show object labels in plots
        show_conf=True,                    # Show object confidence scores in plots
        show_boxes=True,                   # Show object bounding boxes
        line_width=None,                   # Line width of bounding boxes
        format='torchscript',              # Format to export to
        keras=False,                       # Use Keras
        optimize=False,                    # TorchScript: optimize for mobile
        int8=False,                        # CoreML/TF INT8 quantization
        dynamic=False,                     # ONNX/TF/TensorRT: dynamic axes
        simplify=False,                    # ONNX: simplify model
        opset=None,                        # ONNX: opset version
        workspace=4,                       # TensorRT: workspace size (GB)
        nms=False,                         # CoreML: add NMS
        patience=50,                       # Early stopping patience
        save_period=10,                    # Save checkpoint every N epochs
        cache=False,                       # Cache images for faster training
        rect=False,                        # Rectangular training
        cos_lr=False,                      # Cosine LR scheduler
        close_mosaic=10,                   # Disable mosaic augmentation for final N epochs
        resume=False,                      # Resume training from last checkpoint
        amp=True,                          # Automatic Mixed Precision (AMP) training
        fraction=1.0,                      # Dataset fraction to train on
        profile=False,                     # Profile ONNX and TensorRT speeds during training
        freeze=None,                       # Freeze layers: backbone=10, first3=0 1 2
        multi_scale=False,                 # Multi-scale training
        single_cls=False,                  # Train as single-class dataset
        verbose=True,                      # Verbose output
        seed=0,                            # Random seed for reproducibility
        deterministic=True,                # Deterministic training
    )
    
    training_time = time.time() - start_time
    
    print("\\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Results saved to: {results.save_dir}")
    
    # Print training results
    if hasattr(results, 'best_fitness'):
        print(f"Best mAP50: {results.best_fitness:.4f}")
    if hasattr(results, 'best_epoch'):
        print(f"Best epoch: {results.best_epoch}")
    
    # Validate the model
    print("\\nRunning final validation...")
    try:
        model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        best_model = YOLO(model_path)
        metrics = best_model.val()
        
        print(f"Final Validation Results:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
    except Exception as e:
        print(f"Validation error: {e}")
    
    print("\\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: YOLO12n")
    print(f"Dataset: New Bottle/Cup Dataset")
    print(f"Classes: bottle, cup")
    print(f"Epochs: 100")
    print(f"Device: {device}")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print(f"Last model: {results.save_dir}/weights/last.pt")
    print("=" * 60)
    
    print("\\nNext steps:")
    print("1. Test the model: python test_new_model.py")
    print("2. Run real-time detection: python realtime_new_detection.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
