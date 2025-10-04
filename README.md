# YOLO Data Augmentation Tool

A comprehensive Python script for augmenting YOLO format datasets using the albumentations library. This tool creates multiple augmented versions of images while correctly transforming bounding box coordinates.

## Features

- **Comprehensive Augmentation Pipeline**: Includes geometric, photometric, and filter-based transformations
- **YOLO Format Support**: Correctly handles YOLO bounding box coordinate transformations
- **Progress Tracking**: Real-time progress bar and detailed completion statistics
- **Easy Start**: Simple script for quick setup and execution
- **Robust Error Handling**: Graceful handling of missing files and augmentation failures

## Augmentation Techniques

The script includes the following augmentation techniques:

### Geometric Transformations
- Rotation (up to 20 degrees)
- Horizontal and vertical flipping
- Shift, scale, and rotate combinations

### Photometric Transformations
- Random brightness and contrast adjustments
- Hue, saturation, and value shifts
- Random gamma correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Noise and Blur Effects
- Gaussian noise
- Motion blur
- Gaussian blur

### Weather Effects
- Random rain simulation
- Random fog effects

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Easy Start (Recommended)

For quick setup and execution with default settings:

```bash
python easy_start.py
```

This will:
- Set up the proper directory structure
- Move images to the correct folders
- Create empty label files if needed
- Run augmentation with 5 augmentations per image

### Manual Usage

For more control over the process:

```bash
python augment_data.py --input_dir ./input --output_dir ./output --num_augmentations 10
```

#### Command Line Arguments

- `--input_dir`: Path to the source directory containing images and labels subfolders
- `--output_dir`: Path to the directory where augmented data will be saved
- `--num_augmentations`: Number of augmented versions to create for each original image (default: 5)

## Directory Structure

The script expects the following directory structure:

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

The output will have the same structure:

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

## YOLO Annotation Format

The script expects YOLO format annotation files with the following structure:
```
class_id x_center y_center width height
```

Where all coordinates are normalized (0.0 to 1.0).

## Example Usage

```bash
# Basic usage with 5 augmentations per image
python augment_data.py --input_dir ./dataset --output_dir ./augmented_dataset --num_augmentations 5

# Create 10 augmented versions per image
python augment_data.py --input_dir ./dataset --output_dir ./augmented_dataset --num_augmentations 10

# Easy start with automatic setup
python easy_start.py
```

## Output

The script provides detailed progress information and completion statistics:

```
Found 50 images to process
Creating 5 augmented versions for each image
Total augmented images to be created: 250
--------------------------------------------------
Processing images: 100%|██████████| 50/50 [02:15<00:00, 2.70s/image]

==================================================
AUGMENTATION COMPLETED SUCCESSFULLY!
==================================================
Images processed: 50
Total augmented images created: 250
Expected augmented images: 250
Success rate: 100.0%

Output saved to:
  Images: ./output/images
  Labels: ./output/labels
==================================================
```

## Requirements

- Python 3.7+
- albumentations >= 1.3.0
- opencv-python >= 4.8.0
- tqdm >= 4.65.0
- numpy >= 1.24.0
- Pillow >= 9.5.0

## Notes

- The script automatically creates output directories if they don't exist
- Empty label files are created for images without annotations
- The augmentation pipeline uses random probabilities, so results will vary between runs
- All transformations preserve the YOLO format and coordinate system
- The script includes comprehensive error handling and progress tracking

## Troubleshooting

1. **No images found**: Ensure your input directory has an 'images' subfolder with image files
2. **Permission errors**: Make sure you have write permissions for the output directory
3. **Memory issues**: Reduce the number of augmentations or process images in smaller batches
4. **Missing dependencies**: Run `pip install -r requirements.txt` to install all required packages


