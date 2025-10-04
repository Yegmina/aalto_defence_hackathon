#!/bin/bash
# Transfer script for Raspberry Pi

echo "Transferring files to Raspberry Pi..."

# Replace with your Pi's IP address or hostname
PI_HOST="raspberrypi.local"
PI_USER="pi"
PI_PATH="/home/pi/bottle_detection"

# Create directory on Pi
ssh $PI_USER@$PI_HOST "mkdir -p $PI_PATH"

# Transfer files
scp enhanced_bottle_detection.py $PI_USER@$PI_HOST:$PI_PATH/
scp pi_bottle_detection.py $PI_USER@$PI_HOST:$PI_PATH/
scp setup_raspberry_pi.py $PI_USER@$PI_HOST:$PI_PATH/
scp requirements_yolo.txt $PI_USER@$PI_HOST:$PI_PATH/
scp dataset.yaml $PI_USER@$PI_HOST:$PI_PATH/

# Transfer YOLO model (if exists)
if [ -f "yolo12n.pt" ]; then
    scp yolo12n.pt $PI_USER@$PI_HOST:$PI_PATH/
fi

echo "Files transferred successfully!"
echo "Now SSH into your Pi and run:"
echo "  cd $PI_PATH"
echo "  python3 setup_raspberry_pi.py"
