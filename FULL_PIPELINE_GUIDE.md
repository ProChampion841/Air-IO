# Complete Pipeline: Training to Real-Time Deployment

## Overview
This guide covers the full workflow from preparing your dataset to deploying with real sensors.

```
Your Dataset → Train AirIMU → Train AirIO → Real-Time Deployment
```

---

## Phase 1: Prepare Your Dataset

### Required Files Structure
```
data/YourDataset/
├── train/
│   ├── sequence1/
│   │   ├── imu_data.csv          # Required
│   │   ├── groundTruthPoses.csv  # Required
│   │   └── thrust_data.csv       # Optional (for fixed-wing: single throttle)
│   ├── sequence2/
│   └── ...
└── test/
    ├── sequence1/
    └── ...
```

### File Formats

**imu_data.csv**:
```csv
# timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
1524899787.001822, 0.01, -0.02, 1.25, 0.05, -0.03, 9.81
```

**groundTruthPoses.csv**:
```csv
# timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
1524899787.001822, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
```

**thrust_data.csv** (for fixed-wing with single throttle):
```csv
# timestamp, throttle
1524899787.001822, 0.65
```

---

## Phase 2: Create Dataset Configuration

### Step 1: Create Dataset Class

Create `datasets/YourDataset.py`:

```python
import torch
import numpy as np
import pypose as pp
from datasets.EuRoCdataset import Euroc  # Use as template

class YourDataset(Euroc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self, data_path):
        """Load your dataset format"""
        # Load IMU data
        imu_data = np.loadtxt(f"{data_path}/imu_data.csv", delimiter=',', skiprows=1)
        timestamps = imu_data[:, 0]
        gyro = imu_data[:, 1:4]  # rad/s
        acc = imu_data[:, 4:7]   # m/s^2
        
        # Load ground truth
        gt_data = np.loadtxt(f"{data_path}/groundTruthPoses.csv", delimiter=',', skiprows=1)
        gt_timestamps = gt_data[:, 0]
        gt_pos = gt_data[:, 1:4]
        gt_quat = gt_data[:, 4:8]  # [w, x, y, z]
        
        # Interpolate to common timestamps
        # ... (implement interpolation)
        
        # Convert to required format
        self.data = {
            "dt": np.diff(timestamps, prepend=timestamps[0]),
            "time": timestamps,
            "acc": torch.tensor(acc, dtype=torch.float32),
            "gyro": torch.tensor(gyro, dtype=torch.float32),
            "gt_orientation": pp.SO3(torch.tensor(gt_quat, dtype=torch.float32)),
            "gt_translation": torch.tensor(gt_pos, dtype=torch.float32),
            "velocity": self.compute_velocity(gt_pos, timestamps)
        }
        
        # Apply transformations
        self.set_orientation(self.rot_path, data_path, self.rot_type)
        self.update_coordinate(self.coordinate, self.mode)
        self.remove_gravity(self.remove_g)
```

Register in `datasets/__init__.py`:
```python
from datasets.YourDataset import YourDataset
```

### Step 2: Create Dataset Config

Create `configs/datasets/YourDataset/yourdataset.conf`:

```yaml
train:
{
    mode: train
    coordinate: body_coord
    remove_g: False
    rot_type: None  # Use ground truth during training
    rot_path: None
    data_list:
    [{
        name: YourDataset
        window_size: 1000
        step_size: 10
        data_root: "data/YourDataset"
        data_drive: [sequence1, sequence2, sequence3]
    }]
    gravity: 9.81007
}

test:
{
    mode: test
    coordinate: body_coord
    remove_g: False
    data_list:
    [{
        name: YourDataset
        window_size: 1000
        step_size: 1000
        data_root: "data/YourDataset"
        data_drive: [test_sequence1]
    }]
    gravity: 9.81007
}

inference:
{
    mode: inference
    coordinate: body_coord
    remove_g: False
    rot_type: None  # Will be updated for AirIO
    rot_path: None
    data_list:
    [{
        name: YourDataset
        window_size: 1000
        step_size: 1000
        data_root: "data/YourDataset"
        data_drive: [test_sequence1]
    }]
    gravity: 9.81007
}
```

### Step 3: Create Training Config

Create `configs/YourDataset/motion_body_rot.conf`:

```yaml
dataset:
{
    include "../datasets/YourDataset/yourdataset.conf"
    collate: {type: motion}
}

train:
{
    network: codewithrot  # Use attitude encoding
    lr: 1e-3
    min_lr: 1e-5
    batch_size: 128
    max_epoches: 100
    patience: 5
    factor: 0.2
    weight_decay: 1e-4
}

exp_name: "yourdataset/motion_body_rot"
```

---

## Phase 3: Train AirIMU (Optional but Recommended)

If you want IMU correction, train AirIMU first. Otherwise, skip to Phase 4.

Visit [https://airimu.github.io](https://airimu.github.io/) for AirIMU training instructions.

After training, generate orientation file:
```bash
python evaluation/save_ori.py \
    --dataconf configs/datasets/YourDataset/yourdataset.conf \
    --exp path/to/airimu_net_output.pickle
```

This creates `orientation_output.pickle` with corrected orientations.

---

## Phase 4: Train AirIO

### Train the Model

```bash
python train_motion.py --config configs/YourDataset/motion_body_rot.conf
```

Training outputs:
- `experiments/yourdataset/motion_body_rot/best_model.pth`
- `experiments/yourdataset/motion_body_rot/checkpoint_epoch_XX.pth`
- Training logs in wandb (if enabled)

### Run Inference

```bash
python inference_motion.py --config configs/YourDataset/motion_body_rot.conf
```

Output: `experiments/yourdataset/motion_body_rot/net_output.pickle`

### Evaluate Results

```bash
python evaluation/evaluate_motion.py \
    --dataconf configs/datasets/YourDataset/yourdataset.conf \
    --exp experiments/yourdataset/motion_body_rot \
    --seqlen 500
```

---

## Phase 5: Real-Time Deployment with Real Sensor

### Step 1: Create Sensor Interface

Create `realtime_sensor_interface.py`:

```python
import serial
import numpy as np
import time

class RealIMUSensor:
    """Interface for real IMU sensor"""
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, imu_rate=200):
        self.port = port
        self.baudrate = baudrate
        self.dt = 1.0 / imu_rate
        self.serial = None
        
    def connect(self):
        """Connect to IMU sensor"""
        self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        time.sleep(2)  # Wait for connection
        print(f"Connected to IMU on {self.port}")
        
    def read(self):
        """Read one IMU sample
        
        Returns:
            dict with keys: 'acc', 'gyro', 'dt', 'timestamp'
        """
        if self.serial is None:
            raise RuntimeError("Sensor not connected. Call connect() first.")
        
        # Read line from serial
        line = self.serial.readline().decode('utf-8').strip()
        
        # Parse format: "timestamp,gx,gy,gz,ax,ay,az"
        try:
            data = [float(x) for x in line.split(',')]
            timestamp = data[0]
            gyro = np.array(data[1:4])  # rad/s
            acc = np.array(data[4:7])   # m/s^2
            
            return {
                'acc': acc,
                'gyro': gyro,
                'dt': self.dt,
                'timestamp': timestamp
            }
        except:
            return None
    
    def disconnect(self):
        """Disconnect from sensor"""
        if self.serial:
            self.serial.close()
            print("Disconnected from IMU")
```

### Step 2: Create Real-Time Runner

Create `realtime_deployment.py`:

```python
import os
import sys
import torch
import numpy as np
import pypose as pp
from collections import deque
import time

sys.path.append(os.path.dirname(__file__))

from realtime_sensor_interface import RealIMUSensor
from realtime_pipeline_demo import RealtimeOdometryPipeline

def run_realtime_with_sensor(
    sensor_port='/dev/ttyUSB0',
    airimu_model='experiments/yourdataset/airimu/best_model.pth',
    airio_model='experiments/yourdataset/motion_body_rot/best_model.pth',
    duration=60.0,
    save_results=True
):
    """Run real-time odometry with real IMU sensor"""
    
    print("="*70)
    print("Real-Time IMU Odometry with Real Sensor")
    print("="*70)
    
    # Initialize sensor
    print("\n[1/4] Connecting to IMU sensor...")
    sensor = RealIMUSensor(port=sensor_port, baudrate=115200, imu_rate=200)
    sensor.connect()
    
    # Initialize pipeline
    print("\n[2/4] Loading models...")
    pipeline = RealtimeOdometryPipeline(
        window_size=200,
        airimu_model_path=airimu_model,
        airio_model_path=airio_model,
        use_mock=False  # Use real models
    )
    
    # Initialize state (stationary start)
    print("\n[3/4] Initializing state...")
    init_pos = np.array([0.0, 0.0, 0.0])
    init_vel = np.array([0.0, 0.0, 0.0])
    init_rot = pp.identity_SO3(1)
    pipeline.initialize(init_pos, init_rot, init_vel)
    
    # Run real-time processing
    print("\n[4/4] Processing IMU data in real-time...")
    print("Press Ctrl+C to stop\n")
    
    start_time = time.time()
    sample_count = 0
    results = []
    
    try:
        while (time.time() - start_time) < duration:
            # Read IMU
            imu_data = sensor.read()
            if imu_data is None:
                continue
            
            # Process
            result = pipeline.process_imu(imu_data)
            
            # Display
            sample_count += 1
            if sample_count % 20 == 0:  # Update every 0.1s
                print(f"[{time.time()-start_time:6.2f}s] "
                      f"Pos: [{result['position'][0]:6.2f}, {result['position'][1]:6.2f}, {result['position'][2]:6.2f}] "
                      f"Vel: [{result['velocity'][0]:5.2f}, {result['velocity'][1]:5.2f}, {result['velocity'][2]:5.2f}]")
            
            # Save
            if save_results:
                results.append({
                    'timestamp': imu_data['timestamp'],
                    'position': result['position'].copy(),
                    'velocity': result['velocity'].copy(),
                    'orientation': result['orientation'].matrix().numpy()
                })
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        sensor.disconnect()
    
    # Save results
    if save_results and len(results) > 0:
        output_file = f"realtime_results_{int(time.time())}.npy"
        np.save(output_file, results)
        print(f"\nResults saved to {output_file}")
    
    print("\n" + "="*70)
    print(f"Processed {sample_count} samples in {time.time()-start_time:.2f}s")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0',
                        help='Serial port for IMU sensor')
    parser.add_argument('--airimu-model', type=str, required=True,
                        help='Path to AirIMU model')
    parser.add_argument('--airio-model', type=str, required=True,
                        help='Path to AirIO model')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duration in seconds')
    
    args = parser.parse_args()
    
    run_realtime_with_sensor(
        sensor_port=args.port,
        airimu_model=args.airimu_model,
        airio_model=args.airio_model,
        duration=args.duration
    )
```

### Step 3: Deploy

```bash
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --airimu-model experiments/yourdataset/airimu/best_model.pth \
    --airio-model experiments/yourdataset/motion_body_rot/best_model.pth \
    --duration 60
```

---

## Phase 6: Sensor Data Format

Your IMU sensor should output data via serial in this format:

```
timestamp,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z
1234567890.123,0.01,-0.02,1.25,0.05,-0.03,9.81
```

### Arduino Example

```cpp
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  mpu.initialize();
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // Convert to SI units
  float acc_x = ax / 16384.0 * 9.81;  // m/s^2
  float acc_y = ay / 16384.0 * 9.81;
  float acc_z = az / 16384.0 * 9.81;
  float gyro_x = gx / 131.0 * 0.0174533;  // rad/s
  float gyro_y = gy / 131.0 * 0.0174533;
  float gyro_z = gz / 131.0 * 0.0174533;
  
  // Output format
  Serial.print(millis() / 1000.0, 6);
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
  
  delay(5);  // 200Hz
}
```

---

## Quick Reference Commands

```bash
# 1. Train AirIO
python train_motion.py --config configs/YourDataset/motion_body_rot.conf

# 2. Inference
python inference_motion.py --config configs/YourDataset/motion_body_rot.conf

# 3. Evaluate
python evaluation/evaluate_motion.py \
    --dataconf configs/datasets/YourDataset/yourdataset.conf \
    --exp experiments/yourdataset/motion_body_rot \
    --seqlen 500

# 4. Real-time deployment
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --airimu-model experiments/yourdataset/airimu/best_model.pth \
    --airio-model experiments/yourdataset/motion_body_rot/best_model.pth \
    --duration 60
```

---

## Troubleshooting

### Issue: Poor training performance
- Check dataset quality (ground truth accuracy)
- Increase training epochs
- Adjust learning rate
- Try different window sizes

### Issue: Real-time latency
- Reduce window size (e.g., 100 instead of 200)
- Use GPU for inference
- Optimize network architecture

### Issue: Sensor connection failed
- Check port name: `ls /dev/tty*` (Linux) or Device Manager (Windows)
- Verify baudrate matches sensor
- Check USB cable and permissions

### Issue: Drift in real-time
- Calibrate IMU biases before start
- Tune EKF Q and R matrices
- Ensure proper initialization
