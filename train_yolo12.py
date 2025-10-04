#!/usr/bin/env python3
"""
YOLO12 Training Script

This script trains a YOLO12 model on the defence hackathon dataset.
"""

import torch
from ultralytics import YOLO
import os


def main():
    """Main training function."""
    print("=" * 60)
    print("YOLO12 Training for Defence Hackathon")
    print("=" * 60)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLO12 model
    print("Loading YOLO12 model...")
    model = YOLO('yolo12n.pt')  # You can use yolo12s.pt, yolo12m.pt, yolo12l.pt, yolo12x.pt
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data='dataset.yaml',           # Dataset configuration
        epochs=100,                    # Number of training epochs
        imgsz=640,                     # Input image size
        batch=16,                      # Batch size (adjust based on GPU memory)
        device=device,                 # Device to use
        workers=4,                     # Number of worker threads
        project='runs/train',          # Project directory
        name='yolo12_defence_hackathon',  # Experiment name
        exist_ok=True,                 # Overwrite existing experiment
        pretrained=True,               # Use pretrained weights
        optimizer='AdamW',             # Optimizer
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate (lr0 * lrf)
        momentum=0.937,                # SGD momentum
        weight_decay=0.0005,           # Optimizer weight decay
        warmup_epochs=3,               # Warmup epochs
        warmup_momentum=0.8,           # Warmup initial momentum
        warmup_bias_lr=0.1,            # Warmup initial bias lr
        box=7.5,                       # Box loss gain
        cls=0.5,                       # Cls loss gain
        dfl=1.5,                       # Dfl loss gain
        pose=12.0,                     # Pose loss gain
        kobj=1.0,                      # Keypoint obj loss gain
        label_smoothing=0.0,           # Label smoothing
        nbs=64,                        # Nominal batch size
        overlap_mask=True,             # Masks should overlap during training
        mask_ratio=4,                  # Mask downsample ratio
        dropout=0.0,                   # Use dropout regularization
        val=True,                      # Validate/test during training
        split='val',                   # Dataset split to use for validation
        save_json=False,               # Save results to JSON
        save_hybrid=False,             # Save hybrid version of labels
        conf=None,                     # Object confidence threshold
        iou=0.7,                       # NMS IoU threshold
        max_det=300,                   # Maximum detections per image
        half=False,                    # Use half precision (FP16)
        dnn=False,                     # Use OpenCV DNN for ONNX inference
        plots=True,                    # Save plots during training
        source=None,                   # Source directory for images or videos
        vid_stride=1,                  # Video frame-rate stride
        stream_buffer=False,           # Buffer all streaming images
        visualize=False,               # Visualize model features
        augment=False,                 # Apply image augmentation to prediction sources
        agnostic_nms=False,            # Class-agnostic NMS
        retina_masks=False,            # Use high-resolution segmentation masks
        embed=None,                    # Return feature vectors/embeddings from given layers
        show=False,                    # Show results if possible
        save_frames=False,             # Save video frames
        save_txt=False,                # Save results as .txt file
        save_conf=False,               # Save results with confidence scores
        save_crop=False,               # Save cropped images with results
        show_labels=True,              # Show object labels in plots
        show_conf=True,                # Show object confidence scores in plots
        show_boxes=True,               # Show object bounding boxes
        line_width=None,               # Line width of bounding boxes
        format='torchscript',          # Format to export to
        keras=False,                   # Use Keras
        optimize=False,                # TorchScript: optimize for mobile
        int8=False,                    # CoreML/TF INT8 quantization
        dynamic=False,                 # ONNX/TF/TensorRT: dynamic axes
        simplify=False,                # ONNX: simplify model
        opset=None,                    # ONNX: opset version
        workspace=4,                   # TensorRT: workspace size (GB)
        nms=False,                     # CoreML: add NMS
    )
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    
    # Print training results
    print("\nTraining Results:")
    print(f"Best mAP50: {results.best_fitness:.4f}")
    print(f"Best epoch: {results.best_epoch}")
    
    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
