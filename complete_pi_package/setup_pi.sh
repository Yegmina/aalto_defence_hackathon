#!/bin/bash
echo "Setting up Raspberry Pi for Bottle Detection..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git

# Install Python packages
pip3 install ultralytics opencv-python-headless numpy Pillow PyYAML matplotlib seaborn pandas tqdm psutil

# Make scripts executable
chmod +x *.py

echo "Setup complete!"
echo "To run detection: python3 pi_bottle_detection.py"
