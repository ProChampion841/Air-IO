# Quick Start: Training to Real-Time Deployment

## Complete Workflow in 5 Steps

### Step 1: Prepare Your Dataset (5 minutes)

```bash
# Create dataset structure
mkdir -p data/MyDataset/train/flight1
mkdir -p data/MyDataset/test/flight1

# Place your files:
# - imu_data.csv (timestamp, gx, gy, gz, ax, ay, az)
# - groundTruthPoses.csv (timestamp, px, py, pz, qw, qx, qy, qz)
```

### Step 2: Configure Dataset (10 minutes)

Create `configs/datasets/MyDataset/mydataset.conf`:

```yaml
train:
{
    mode: train
    coordinate: body_coord
    remove_g: False
    data_list:
    [{
        name: YourDataset  # Must match class name in datasets/
        window_size: 1000
        step_size: 10
        data_root: "data/MyDataset"
        data_drive: [flight1]
    }]
    gravity: 9.81007
}
```

Create `configs/MyDataset/motion_body_rot.conf`:

```yaml
dataset:
{
    include "../datasets/MyDataset/mydataset.conf"
    collate: {type: motion}
}

train:
{
    network: codewithrot
    lr: 1e-3
    batch_size: 128
    max_epoches: 100
}

exp_name: "mydataset/motion_body_rot"
```

### Step 3: Train AirIO (2-4 hours)

```bash
# Train
python train_motion.py --config configs/MyDataset/motion_body_rot.conf

# Output: experiments/mydataset/motion_body_rot/best_model.pth
```

### Step 4: Test Sensor Connection (1 minute)

```bash
# Test your IMU sensor
python realtime_sensor_interface.py --port /dev/ttyUSB0 --duration 5

# Expected output:
# ✓ Connected to IMU on /dev/ttyUSB0 @ 115200 baud
# [timestamp] Gyro: [x, y, z] Acc: [x, y, z]
```

### Step 5: Deploy Real-Time (Ready!)

```bash
# Run with your trained model
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --airio-model experiments/mydataset/motion_body_rot/best_model.pth \
    --duration 60 \
    --save \
    --log-imu

# Visualize results
python realtime_deployment.py --visualize realtime_results_YYYYMMDD_HHMMSS.npy
```

---

## Detailed Commands

### Training Commands

```bash
# 1. Train AirIO
python train_motion.py --config configs/MyDataset/motion_body_rot.conf

# 2. Run inference on test set
python inference_motion.py --config configs/MyDataset/motion_body_rot.conf

# 3. Evaluate results
python evaluation/evaluate_motion.py \
    --dataconf configs/datasets/MyDataset/mydataset.conf \
    --exp experiments/mydataset/motion_body_rot \
    --seqlen 500
```

### Real-Time Deployment Commands

```bash
# Test sensor connection
python realtime_sensor_interface.py \
    --port /dev/ttyUSB0 \
    --baudrate 115200 \
    --duration 5

# Run real-time odometry (demo mode - no models)
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --duration 30

# Run with trained models
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --airio-model experiments/mydataset/motion_body_rot/best_model.pth \
    --duration 60 \
    --save \
    --log-imu \
    --display-rate 10

# Visualize saved results
python realtime_deployment.py \
    --visualize realtime_results_20250101_120000.npy
```

---

## IMU Sensor Setup

### Arduino Code (MPU6050 Example)

Save as `imu_streamer.ino`:

```cpp
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;
unsigned long lastTime = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
  
  if (!mpu.testConnection()) {
    Serial.println("# MPU6050 connection failed");
    while(1);
  }
  
  Serial.println("# timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z");
  lastTime = micros();
}

void loop() {
  unsigned long currentTime = micros();
  
  // Read at 200Hz (5000 microseconds)
  if (currentTime - lastTime >= 5000) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // Convert to SI units
    float acc_x = ax / 16384.0 * 9.81;
    float acc_y = ay / 16384.0 * 9.81;
    float acc_z = az / 16384.0 * 9.81;
    float gyro_x = gx / 131.0 * 0.0174533;
    float gyro_y = gy / 131.0 * 0.0174533;
    float gyro_z = gz / 131.0 * 0.0174533;
    
    // Output: timestamp,gx,gy,gz,ax,ay,az
    Serial.print(currentTime / 1000000.0, 6);
    Serial.print(",");
    Serial.print(gyro_x, 6);
    Serial.print(",");
    Serial.print(gyro_y, 6);
    Serial.print(",");
    Serial.print(gyro_z, 6);
    Serial.print(",");
    Serial.print(acc_x, 6);
    Serial.print(",");
    Serial.print(acc_y, 6);
    Serial.print(",");
    Serial.println(acc_z, 6);
    
    lastTime = currentTime;
  }
}
```

### Upload to Arduino

```bash
# Install Arduino CLI
# Upload code
arduino-cli compile --fqbn arduino:avr:uno imu_streamer.ino
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno imu_streamer.ino
```

---

## File Structure After Setup

```
Air-IO/
├── data/
│   └── MyDataset/
│       ├── train/
│       │   └── flight1/
│       │       ├── imu_data.csv
│       │       └── groundTruthPoses.csv
│       └── test/
│           └── flight1/
│               ├── imu_data.csv
│               └── groundTruthPoses.csv
├── configs/
│   ├── datasets/
│   │   └── MyDataset/
│   │       └── mydataset.conf
│   └── MyDataset/
│       └── motion_body_rot.conf
├── experiments/
│   └── mydataset/
│       └── motion_body_rot/
│           ├── best_model.pth
│           └── net_output.pickle
├── realtime_sensor_interface.py
├── realtime_deployment.py
└── realtime_results_YYYYMMDD_HHMMSS.npy
```

---

## Troubleshooting

### Training Issues

**Error: "Dataset class not found"**
```bash
# Make sure class name matches config
# In datasets/YourDataset.py: class YourDataset(Euroc)
# In config: name: YourDataset
```

**Error: "CUDA out of memory"**
```bash
# Reduce batch size in config
train: { batch_size: 64 }  # Instead of 128
```

### Sensor Issues

**Error: "Permission denied: /dev/ttyUSB0"**
```bash
# Linux: Add user to dialout group
sudo usermod -a -G dialout $USER
# Logout and login again
```

**Error: "No data received"**
```bash
# Check baudrate matches Arduino
# Check USB cable
# Check Arduino is running (LED blinking)
# Try: ls /dev/tty* to find correct port
```

### Real-Time Issues

**High latency**
```bash
# Reduce window size (faster but less accurate)
# In realtime_pipeline_demo.py: window_size=100
```

**Drift accumulation**
```bash
# Calibrate IMU before start (keep stationary for 10s)
# Tune EKF Q and R matrices
# Use better ground truth for training
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Training time | 2-4 hours (100 epochs) |
| Inference speed | 200 Hz (real-time) |
| Position error | 0.1-0.5 m (after 60s) |
| Velocity error | 0.05-0.2 m/s |
| Latency | 5-10 ms |

---

## Next Steps

1. **Improve accuracy**: Train AirIMU for IMU correction
2. **Add visualization**: Real-time 3D trajectory display
3. **Optimize**: Use TensorRT for faster inference
4. **Deploy**: Integrate with your robot/drone control system

For detailed information, see `FULL_PIPELINE_GUIDE.md`
