# Real-Time IMU Odometry Deployment Guide

## Can You Use This Pipeline Real-Time?

**YES**, but requires modifications. The current code is **offline** (batch processing). Here's how to adapt it for **real-time** use.

---

## Current vs Real-Time Comparison

| Aspect | Current (Offline) | Real-Time Needed |
|--------|------------------|------------------|
| **Data Input** | Pre-recorded datasets | Live IMU sensor stream |
| **Processing** | Batch (entire sequence) | Incremental (sample-by-sample) |
| **Network Input** | Fixed windows (1000 frames) | Sliding window buffer |
| **Output** | Saved to files | Immediate return |
| **Latency** | Not critical | Must be < sensor rate |

---

## Real-Time Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  REAL-TIME PIPELINE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  IMU Sensor (200 Hz)                                    │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────┐                                    │
│  │ Sliding Buffer  │  (Store last 1000 samples)        │
│  │  [acc, gyro]    │                                    │
│  └────────┬────────┘                                    │
│           │                                             │
│           ├──────────────┬──────────────┐               │
│           ▼              ▼              ▼               │
│      Every Sample   Every N samples  Every N samples   │
│           │              │              │               │
│           │         ┌────▼─────┐   ┌───▼────┐          │
│           │         │ AirIMU   │   │ AirIO  │          │
│           │         │ Network  │   │ Network│          │
│           │         └────┬─────┘   └───┬────┘          │
│           │              │             │                │
│           │              ▼             ▼                │
│           │         Corrected IMU  Velocity            │
│           │              │             │                │
│           └──────────────┴─────────────┘                │
│                          │                              │
│                          ▼                              │
│                   ┌──────────────┐                      │
│                   │     EKF      │  (Every sample)     │
│                   │  Propagate + │                      │
│                   │    Update    │                      │
│                   └──────┬───────┘                      │
│                          │                              │
│                          ▼                              │
│                   ┌──────────────┐                      │
│                   │   Odometry   │  (R, V, P)          │
│                   │    Output    │                      │
│                   └──────────────┘                      │
│                          │                              │
│                          ▼                              │
│                  Application (Navigation, Control)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Load Pre-trained Models

```python
import torch
from model import CodeNetMotionwithRot  # Your AirIO network
# Assume AirIMU model is loaded similarly

# Load AirIO model
airio_checkpoint = torch.load('experiments/euroc/motion_body_rot/ckpt/best_model.ckpt')
airio_model = CodeNetMotionwithRot(config)
airio_model.load_state_dict(airio_checkpoint['model_state_dict'])
airio_model.eval()

# Load AirIMU model (from separate project)
airimu_checkpoint = torch.load('AirIMU_Model_Results/AirIMU_EuRoC/best_model.ckpt')
airimu_model = AirIMUNetwork(config)
airimu_model.load_state_dict(airimu_checkpoint['model_state_dict'])
airimu_model.eval()
```

### Step 2: Initialize Real-Time System

```python
from EKF.IMUrealtimerunner import RealtimeOdometry

# Create odometry system
odometry = RealtimeOdometry(
    airimu_model=airimu_model,
    airio_model=airio_model,
    window_size=1000,  # Must match training window
    device='cuda'  # or 'cpu'
)

# Initialize with known starting state
odometry.initialize(
    init_pos=[0.0, 0.0, 0.0],      # Starting position
    init_rot=pp.identity_SO3(1),    # Starting orientation
    init_vel=[0.0, 0.0, 0.0],       # Starting velocity
    gravity=9.81007
)
```

### Step 3: Connect to IMU Sensor

#### Option A: ROS Integration
```python
import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class RealtimeOdometryNode:
    def __init__(self):
        self.odometry = RealtimeOdometry(airimu_model, airio_model)
        self.odometry.initialize([0,0,0], pp.identity_SO3(1), [0,0,0])
        
        self.last_time = None
        self.sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        self.pub = rospy.Publisher('/odometry', Odometry, queue_size=10)
    
    def imu_callback(self, msg):
        # Extract IMU data
        acc = [msg.linear_acceleration.x, 
               msg.linear_acceleration.y, 
               msg.linear_acceleration.z]
        gyro = [msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z]
        
        # Calculate dt
        current_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Process IMU
        result = self.odometry.process_imu(acc, gyro, dt)
        
        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.pose.pose.position.x = result['position'][0]
        odom_msg.pose.pose.position.y = result['position'][1]
        odom_msg.pose.pose.position.z = result['position'][2]
        # ... set orientation and velocity
        self.pub.publish(odom_msg)

if __name__ == '__main__':
    rospy.init_node('airio_odometry')
    node = RealtimeOdometryNode()
    rospy.spin()
```

#### Option B: Direct Sensor (PySerial, etc.)
```python
import serial
import struct

# Connect to IMU sensor
ser = serial.Serial('/dev/ttyUSB0', 115200)

last_time = time.time()
while True:
    # Read IMU packet (format depends on your sensor)
    data = ser.read(28)  # Example: 7 floats * 4 bytes
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, timestamp = struct.unpack('7f', data)
    
    # Calculate dt
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    
    # Process
    result = odometry.process_imu(
        acc=[acc_x, acc_y, acc_z],
        gyro=[gyro_x, gyro_y, gyro_z],
        dt=dt
    )
    
    print(f"Position: {result['position']}")
```

#### Option C: Simulation/Replay
```python
import pandas as pd

# Load recorded IMU data
df = pd.read_csv('imu_data.csv')

for i in range(len(df)):
    acc = [df.loc[i, 'acc_x'], df.loc[i, 'acc_y'], df.loc[i, 'acc_z']]
    gyro = [df.loc[i, 'gyro_x'], df.loc[i, 'gyro_y'], df.loc[i, 'gyro_z']]
    dt = df.loc[i, 'dt']
    
    result = odometry.process_imu(acc, gyro, dt)
    
    # Visualize or save
    trajectory.append(result['position'])
```

---

## Performance Considerations

### 1. Latency Requirements

| Component | Typical Latency | Optimization |
|-----------|----------------|--------------|
| **AirIMU Network** | 5-10 ms | Run every N samples (e.g., 10) |
| **AirIO Network** | 5-10 ms | Run every N samples (e.g., 10) |
| **EKF Update** | <1 ms | Run every sample |
| **Total** | 10-20 ms | Acceptable for 200 Hz IMU |

### 2. Optimization Strategies

#### Strategy 1: Reduce Network Frequency
```python
class RealtimeOdometry:
    def __init__(self, ..., network_update_rate=10):
        self.network_update_rate = network_update_rate
        self.sample_count = 0
    
    def process_imu(self, acc, gyro, dt):
        self.sample_count += 1
        
        # Run networks every N samples
        velocity_obs = None
        if self.sample_count % self.network_update_rate == 0:
            velocity_obs = self._run_networks()
        
        # EKF runs every sample (fast)
        # ... rest of code
```

#### Strategy 2: Use Smaller Window
```python
# Trade accuracy for speed
odometry = RealtimeOdometry(
    airimu_model, airio_model,
    window_size=500  # Instead of 1000
)
```

#### Strategy 3: GPU Acceleration
```python
# Use CUDA for network inference
odometry = RealtimeOdometry(
    airimu_model, airio_model,
    device='cuda'  # Much faster than CPU
)
```

#### Strategy 4: Model Quantization
```python
# Quantize models for faster inference
airio_model = torch.quantization.quantize_dynamic(
    airio_model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## Key Differences from Offline

### Offline (Current Code)
```python
# Loads entire sequence
inference_state_load = pickle.load('net_output.pickle')

# Processes all at once
for data in dataset:
    # Process entire sequence
    pass
```

### Real-Time (Modified)
```python
# Maintains sliding buffer
buffer = deque(maxlen=1000)

# Processes incrementally
def imu_callback(acc, gyro, dt):
    buffer.append({'acc': acc, 'gyro': gyro})
    
    if len(buffer) >= window_size:
        # Run network on buffer
        velocity = network(buffer)
    
    # EKF update (always)
    state = ekf.update(acc, gyro, velocity, dt)
    return state
```

---

## Challenges & Solutions

### Challenge 1: Network Requires Sequences
**Problem**: Networks trained on 1000-frame windows, but real-time has 1 frame at a time

**Solution**: Maintain sliding window buffer
```python
self.imu_buffer = deque(maxlen=1000)
# Add new sample, automatically removes oldest
self.imu_buffer.append(new_sample)
```

### Challenge 2: Initial Buffer Filling
**Problem**: First 1000 samples have no network prediction

**Solution**: Use EKF propagation only
```python
if len(buffer) < window_size:
    # Propagate only (no velocity measurement)
    state = ekf.propagate(imu_data)
else:
    # Propagate + Update with velocity
    velocity = network(buffer)
    state = ekf.update(imu_data, velocity)
```

### Challenge 3: Computational Load
**Problem**: Running networks at 200 Hz is expensive

**Solution**: Run networks at lower rate (e.g., 20 Hz)
```python
# Network: 20 Hz (every 10 samples)
# EKF: 200 Hz (every sample)
if sample_count % 10 == 0:
    velocity = network(buffer)
ekf.update(imu, velocity)  # velocity=None if not updated
```

### Challenge 4: Model Loading Time
**Problem**: Loading models takes time at startup

**Solution**: Pre-load and keep in memory
```python
# Load once at initialization
models = load_models()  # Takes 1-2 seconds

# Then use for entire session
while running:
    result = process_imu(models, imu_data)
```

---

## Testing Real-Time Performance

### Benchmark Script
```python
import time

# Measure latency
latencies = []
for i in range(1000):
    start = time.time()
    result = odometry.process_imu(acc, gyro, dt)
    latency = time.time() - start
    latencies.append(latency)

print(f"Mean latency: {np.mean(latencies)*1000:.2f} ms")
print(f"Max latency: {np.max(latencies)*1000:.2f} ms")
print(f"99th percentile: {np.percentile(latencies, 99)*1000:.2f} ms")

# Check if real-time capable
imu_rate = 200  # Hz
max_allowed_latency = 1.0 / imu_rate  # 5 ms for 200 Hz
if np.max(latencies) < max_allowed_latency:
    print("✓ Real-time capable!")
else:
    print("✗ Too slow for real-time")
```

---

## Deployment Checklist

- [ ] Load pre-trained AirIMU and AirIO models
- [ ] Initialize RealtimeOdometry with correct window size
- [ ] Set initial state (position, orientation, velocity)
- [ ] Connect to IMU sensor (ROS, serial, etc.)
- [ ] Implement callback to process each IMU sample
- [ ] Test latency meets real-time requirements (<5ms for 200Hz)
- [ ] Handle edge cases (buffer filling, sensor dropout)
- [ ] Implement output interface (publish odometry, save trajectory)
- [ ] Add error handling and recovery
- [ ] Monitor performance in production

---

## Example: Complete Real-Time System

```python
#!/usr/bin/env python3
"""Complete real-time IMU odometry system"""

import torch
import pypose as pp
from EKF.IMUrealtimerunner import RealtimeOdometry
from model import CodeNetMotionwithRot, AirIMUNetwork

def main():
    # 1. Load models
    print("Loading models...")
    airio_model = torch.load('path/to/airio_best_model.ckpt')
    airimu_model = torch.load('path/to/airimu_best_model.ckpt')
    
    # 2. Initialize odometry
    print("Initializing odometry...")
    odometry = RealtimeOdometry(
        airimu_model, airio_model,
        window_size=1000,
        device='cuda'
    )
    odometry.initialize([0,0,0], pp.identity_SO3(1), [0,0,0])
    
    # 3. Connect to sensor
    print("Connecting to IMU sensor...")
    import rospy
    from sensor_msgs.msg import Imu
    
    def imu_callback(msg):
        acc = [msg.linear_acceleration.x, 
               msg.linear_acceleration.y, 
               msg.linear_acceleration.z]
        gyro = [msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z]
        dt = 0.005  # 200 Hz
        
        result = odometry.process_imu(acc, gyro, dt)
        print(f"Pos: {result['position']}, Vel: {result['velocity']}")
    
    rospy.init_node('airio_realtime')
    rospy.Subscriber('/imu/data', Imu, imu_callback)
    print("Real-time odometry running...")
    rospy.spin()

if __name__ == '__main__':
    main()
```

---

## Summary

**YES, you can use this for real-time**, but:

1. ✅ **EKF is already real-time ready** - processes sample-by-sample
2. ⚠️ **Networks need adaptation** - use sliding window buffer
3. ⚠️ **Reduce network frequency** - run every N samples (e.g., 10)
4. ✅ **Use provided `IMUrealtimerunner.py`** - handles all adaptations
5. ⚠️ **Test latency** - ensure <5ms for 200Hz IMU

**Recommended Setup**:
- Network inference: 20 Hz (every 10 samples)
- EKF update: 200 Hz (every sample)
- Expected latency: 10-20 ms
- Works with 200 Hz IMU sensors
