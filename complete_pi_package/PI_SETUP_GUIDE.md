# Raspberry Pi 4 Setup Guide for Bottle Detection

## Prerequisites

1. **Raspberry Pi 4 Model B** with SD card (8GB+ recommended)
2. **Raspbian OS** installed and updated
3. **Network connection** (WiFi or Ethernet)
4. **USB Camera** or **Raspberry Pi Camera Module**
5. **SSH access** enabled

## Step 1: Prepare Raspberry Pi

### Enable SSH (if not already enabled)
```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Update the system
```bash
sudo apt update
sudo apt upgrade -y
```

### Install Python and basic tools
```bash
sudo apt install -y python3 python3-pip python3-venv git
```

## Step 2: Transfer Files from Laptop

### Option A: Using SCP (Recommended)
1. **Find your Pi's IP address:**
   ```bash
   # On Pi, run:
   hostname -I
   ```

2. **Transfer files from laptop:**
   ```bash
   # On Windows, run:
   transfer_to_pi.bat
   
   # Or manually:
   scp enhanced_bottle_detection.py pi@raspberrypi.local:/home/pi/bottle_detection/
   scp pi_bottle_detection.py pi@raspberrypi.local:/home/pi/bottle_detection/
   scp setup_raspberry_pi.py pi@raspberrypi.local:/home/pi/bottle_detection/
   scp requirements_yolo.txt pi@raspberrypi.local:/home/pi/bottle_detection/
   scp dataset.yaml pi@raspberrypi.local:/home/pi/bottle_detection/
   scp yolo12n.pt pi@raspberrypi.local:/home/pi/bottle_detection/
   ```

### Option B: Using USB Drive
1. Copy all files to USB drive
2. Insert USB drive into Pi
3. Copy files to Pi:
   ```bash
   mkdir -p /home/pi/bottle_detection
   cp /media/pi/USB_DRIVE_NAME/* /home/pi/bottle_detection/
   ```

## Step 3: Setup on Raspberry Pi

### SSH into your Pi
```bash
ssh pi@raspberrypi.local
# or
ssh pi@YOUR_PI_IP_ADDRESS
```

### Navigate to the project directory
```bash
cd /home/pi/bottle_detection
```

### Run the setup script
```bash
python3 setup_raspberry_pi.py
```

This will:
- Install all required system packages
- Install Python dependencies
- Create optimized detection script
- Configure the environment

## Step 4: Test Camera

### Test USB camera
```bash
# Test if camera is detected
ls /dev/video*

# Test camera with fswebcam
sudo apt install fswebcam
fswebcam test.jpg
```

### Test Raspberry Pi Camera Module (if using)
```bash
# Enable camera module
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable

# Test camera
raspistill -o test.jpg
```

## Step 5: Run Bottle Detection

### Start the detection system
```bash
python3 pi_bottle_detection.py
```

### Controls
- **'q'** - Quit detection
- **'s'** - Save screenshot
- **'c'** - Switch camera

## Step 6: Optimize Performance (Optional)

### Increase GPU memory split
```bash
sudo raspi-config
# Navigate to: Advanced Options > Memory Split
# Set to 128 or 256
```

### Overclock (if needed)
```bash
sudo raspi-config
# Navigate to: Advanced Options > Overclock
# Choose appropriate setting
```

## Troubleshooting

### Camera not detected
```bash
# Check USB devices
lsusb

# Check video devices
ls /dev/video*

# Test with different camera index
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Performance issues
- Reduce resolution in `pi_bottle_detection.py`
- Increase confidence threshold
- Use YOLO11n instead of YOLO12n
- Close unnecessary applications

### Memory issues
```bash
# Check memory usage
free -h

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## File Structure on Pi

```
/home/pi/bottle_detection/
├── pi_bottle_detection.py      # Main detection script
├── enhanced_bottle_detection.py # Original script
├── setup_raspberry_pi.py       # Setup script
├── requirements_yolo.txt       # Python dependencies
├── dataset.yaml               # Dataset configuration
├── yolo12n.pt                 # YOLO model
└── pi_detection_output/       # Output directory
    ├── pi_2025-01-04_14.30_00001.jpg
    ├── pi_2025-01-04_14.30_00001.json
    └── ...
```

## Performance Expectations

- **Resolution:** 320x240 (optimized for Pi)
- **FPS:** 10-15 FPS
- **Detection:** Real-time bottle/cup detection
- **Output:** Images and JSON data saved automatically

## Network Setup (Optional)

### Set static IP (recommended)
```bash
sudo nano /etc/dhcpcd.conf
# Add:
interface eth0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1
```

### Enable VNC for remote desktop
```bash
sudo raspi-config
# Navigate to: Interface Options > VNC > Enable
```

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify camera connection and permissions
3. Ensure all dependencies are installed
4. Check Pi's temperature and performance
5. Review log output for specific error messages
