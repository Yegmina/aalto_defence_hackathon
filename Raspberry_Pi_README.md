# Raspberry Pi Deployment Module

This module provides comprehensive support for deploying the YOLO12 object detection system on Raspberry Pi devices. It includes optimized detection scripts, automated setup utilities, and deployment guides for edge computing applications.

## Files Overview

### Pi-optimized Scripts
- **`pi_bottle_detection.py`**: Lightweight detection script for Pi devices
- **`setup_raspberry_pi.py`**: Automated Pi setup and configuration
- **`transfer_to_pi.sh`**: Automated code transfer to Pi devices

### Complete Deployment Package
- **`complete_pi_package/`**: Self-contained deployment package
  - `pi_bottle_detection.py`: Optimized detection script
  - `enhanced_bottle_detection.py`: Full-featured detection
  - `dataset.yaml`: Dataset configuration
  - `requirements_yolo.txt`: Dependencies
  - `setup_pi.sh`: Setup script
  - `PI_SETUP_GUIDE.md`: Comprehensive setup guide
  - `wpa_supplicant.conf`: WiFi configuration template
  - `ssh`: SSH enable file

## Features

### Pi-optimized Performance
- **Lightweight Model**: Uses YOLO12n (nano) for faster inference
- **Reduced Resolution**: 320x240 input for better performance
- **Memory Optimization**: Efficient memory usage for 4GB Pi
- **CPU Optimization**: Multi-threading for better CPU utilization

### Automated Setup
- **One-click Installation**: Automated dependency installation
- **Network Configuration**: WiFi and SSH setup automation
- **System Optimization**: Pi-specific performance tuning
- **Error Handling**: Comprehensive error checking and recovery

### Deployment Utilities
- **Code Transfer**: Automated file transfer to Pi devices
- **Remote Execution**: SSH-based remote script execution
- **Configuration Management**: Centralized configuration handling
- **Monitoring**: System resource monitoring and logging

## Hardware Requirements

### Minimum Requirements
- **Raspberry Pi**: Pi 4 Model B (4GB RAM recommended)
- **Storage**: 32GB+ microSD card (Class 10 or better)
- **Camera**: Raspberry Pi Camera Module v2 or USB webcam
- **Power**: Official Pi power supply (5V 3A)

### Recommended Configuration
- **Raspberry Pi**: Pi 4 Model B (8GB RAM)
- **Storage**: 64GB+ microSD card (UHS-I Class 10)
- **Camera**: Raspberry Pi Camera Module v3
- **Cooling**: Active cooling solution for sustained performance
- **Case**: Proper ventilation for thermal management

## Setup Process

### Phase 1: Pi Preparation
1. **OS Installation**: Flash Raspberry Pi OS to microSD card
2. **Initial Setup**: Configure basic Pi settings
3. **Network Setup**: Configure WiFi and SSH access
4. **System Update**: Update system packages and firmware

### Phase 2: Software Installation
1. **Python Setup**: Install Python 3.8+ and pip
2. **Dependencies**: Install required packages
3. **Camera Setup**: Configure camera module
4. **Performance Tuning**: Optimize Pi for detection tasks

### Phase 3: Code Deployment
1. **File Transfer**: Copy detection scripts to Pi
2. **Model Deployment**: Transfer trained model files
3. **Configuration**: Set up dataset and model paths
4. **Testing**: Verify detection system functionality

## Usage

### Quick Setup
```bash
# Run automated setup
python setup_raspberry_pi.py

# Or use the complete package
cd complete_pi_package
bash setup_pi.sh
```

### Basic Detection
```bash
# Run Pi-optimized detection
python pi_bottle_detection.py
```

### Advanced Detection
```bash
# Run full-featured detection
python enhanced_bottle_detection.py
```

## Performance Optimization

### System Optimization
- **GPU Memory Split**: Allocate 128MB to GPU
- **CPU Governor**: Set to performance mode
- **Thermal Management**: Enable thermal throttling protection
- **Memory Management**: Optimize swap usage

### Detection Optimization
- **Model Size**: Use YOLO12n for best performance
- **Input Resolution**: 320x240 for 15+ FPS
- **Batch Size**: Single image processing
- **Threading**: Use multiple threads for I/O

### Performance Targets
- **FPS**: 10-15 FPS (Pi 4 with 4GB RAM)
- **Latency**: <100ms end-to-end processing
- **Memory**: <2GB RAM usage
- **CPU**: <80% CPU utilization

## Configuration

### Pi-specific Settings
```python
# Pi optimization settings
PI_OPTIMIZED = True
MODEL_SIZE = 'nano'  # Use YOLO12n
INPUT_SIZE = (320, 240)  # Reduced resolution
BATCH_SIZE = 1  # Single image processing
THREADS = 4  # Multi-threading
```

### Camera Configuration
```python
# Pi camera settings
CAMERA_SETTINGS = {
    'resolution': (320, 240),
    'framerate': 30,
    'sensor_mode': 0,
    'exposure_mode': 'auto',
    'awb_mode': 'auto'
}
```

### Network Configuration
```bash
# WiFi configuration (wpa_supplicant.conf)
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="YourWiFiName"
    psk="YourWiFiPassword"
    key_mgmt=WPA-PSK
}
```

## Deployment Methods

### Method 1: Direct Transfer
```bash
# Copy files to Pi
scp -r detection_scripts/ pi@raspberrypi.local:/home/pi/

# SSH into Pi
ssh pi@raspberrypi.local

# Run detection
cd detection_scripts
python pi_bottle_detection.py
```

### Method 2: Automated Transfer
```bash
# Use transfer script
bash transfer_to_pi.sh

# Script handles:
# - File transfer
# - Dependency installation
# - Configuration setup
# - Testing
```

### Method 3: Complete Package
```bash
# Deploy complete package
cd complete_pi_package
bash deploy_to_pi.sh pi@raspberrypi.local
```

## Monitoring and Logging

### System Monitoring
```python
# Monitor system resources
import psutil

cpu_percent = psutil.cpu_percent()
memory_percent = psutil.virtual_memory().percent
temperature = psutil.sensors_temperatures()

print(f"CPU: {cpu_percent}%")
print(f"Memory: {memory_percent}%")
print(f"Temperature: {temperature}°C")
```

### Detection Logging
```python
# Log detection results
import logging

logging.basicConfig(
    filename='detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Detection: {class_name} at {confidence:.2f}")
```

## Troubleshooting

### Common Issues
1. **Low FPS**: Reduce resolution or use smaller model
2. **Memory errors**: Optimize memory usage or add swap
3. **Camera not detected**: Check camera connection and drivers
4. **Thermal throttling**: Improve cooling or reduce processing load

### Performance Issues
1. **High CPU usage**: Enable GPU acceleration or reduce processing
2. **Memory leaks**: Check for proper resource cleanup
3. **Slow startup**: Preload model or use model caching
4. **Inconsistent FPS**: Optimize processing pipeline

### Hardware Issues
1. **Camera not working**: Check camera module connection
2. **Power issues**: Use official Pi power supply
3. **Thermal issues**: Improve cooling and ventilation
4. **SD card issues**: Use high-quality SD card

## Remote Management

### SSH Access
```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Connect remotely
ssh pi@raspberrypi.local
```

### Remote Execution
```bash
# Run detection remotely
ssh pi@raspberrypi.local "cd detection_scripts && python pi_bottle_detection.py"

# Monitor system
ssh pi@raspberrypi.local "htop"
```

### File Transfer
```bash
# Copy files to Pi
scp file.py pi@raspberrypi.local:/home/pi/

# Copy files from Pi
scp pi@raspberrypi.local:/home/pi/results.txt ./
```

## Security Considerations

### Network Security
- **Change default password**: Set strong password for pi user
- **SSH key authentication**: Use SSH keys instead of passwords
- **Firewall configuration**: Enable and configure firewall
- **Regular updates**: Keep system and packages updated

### Access Control
- **User permissions**: Limit user access to necessary functions
- **File permissions**: Set appropriate file permissions
- **Service management**: Control which services are running
- **Log monitoring**: Monitor system and application logs

## Integration with Main System

### Model Compatibility
- **Model Format**: Use PyTorch (.pt) format for Pi
- **Model Size**: Optimize model size for Pi memory
- **Quantization**: Consider model quantization for better performance
- **Pruning**: Remove unnecessary model components

### Data Synchronization
- **Results Transfer**: Sync detection results to main system
- **Model Updates**: Deploy updated models to Pi devices
- **Configuration Sync**: Keep configurations synchronized
- **Log Aggregation**: Collect logs from multiple Pi devices

## Best Practices

### Deployment
- **Test thoroughly**: Validate detection on Pi before deployment
- **Monitor performance**: Track system resources and detection accuracy
- **Backup configurations**: Keep backups of working configurations
- **Document changes**: Record all configuration changes

### Maintenance
- **Regular updates**: Keep system and packages updated
- **Performance monitoring**: Monitor system performance over time
- **Error handling**: Implement robust error handling and recovery
- **Log analysis**: Regularly analyze logs for issues

## Dependencies

### Pi-specific Packages
- `picamera2`: Raspberry Pi camera interface
- `RPi.GPIO`: GPIO control (if needed)
- `adafruit-circuitpython-*`: Hardware interfaces (if needed)

### Standard Packages
- `ultralytics`: YOLO detection framework
- `opencv-python`: Computer vision
- `numpy`: Numerical operations
- `torch`: PyTorch framework
- `Pillow`: Image handling
- `psutil`: System monitoring

### Installation
```bash
# Install Pi-specific packages
pip install picamera2 RPi.GPIO

# Install standard packages
pip install -r requirements_yolo.txt
```

## Support

For Pi deployment issues:
1. Check hardware connections and power supply
2. Verify camera module compatibility and drivers
3. Review system resources and performance
4. Check network connectivity and SSH access
5. Review error logs and system messages
