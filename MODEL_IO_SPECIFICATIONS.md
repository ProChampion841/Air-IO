# AirIO Model Input/Output Specifications

## Overview

The AirIO pipeline consists of two neural networks (AirIMU and AirIO) followed by an EKF. Here are the detailed specifications.

---

## 1. AirIMU Network (IMU Correction)

### Input Features

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **Accelerometer** | `(batch, 200, 3)` | m/s² | Raw acceleration in body frame [ax, ay, az] |
| **Gyroscope** | `(batch, 200, 3)` | rad/s | Raw angular velocity in body frame [ωx, ωy, ωz] |

**Total Input Shape**: `(batch_size, window_size, 6)`
- `batch_size`: Number of sequences (e.g., 128)
- `window_size`: Sequence length (typically 200 samples = 1 second @ 200Hz)
- `6`: 3 acc + 3 gyro

### Output Features

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **Corrected Acc** | `(batch, 200, 3)` | m/s² | Bias-corrected acceleration |
| **Corrected Gyro** | `(batch, 200, 3)` | rad/s | Bias-corrected angular velocity |
| **Orientation** | `(batch, 200, 4)` | quaternion | Estimated orientation [qw, qx, qy, qz] |
| **Acc Covariance** | `(batch, 200, 3)` | m²/s⁴ | Uncertainty of corrected acc |
| **Gyro Covariance** | `(batch, 200, 3)` | rad²/s² | Uncertainty of corrected gyro |

### Example Code

```python
import torch

# Input
acc_raw = torch.randn(128, 200, 3)   # batch=128, window=200, features=3
gyro_raw = torch.randn(128, 200, 3)

# Forward pass
airimu_output = airimu_model(acc_raw, gyro_raw)

# Output
corrected_acc = airimu_output['corrected_acc']      # (128, 200, 3)
corrected_gyro = airimu_output['corrected_gyro']    # (128, 200, 3)
orientation = airimu_output['rotation']             # (128, 200, 4)
acc_cov = airimu_output['acc_cov']                  # (128, 200, 3)
gyro_cov = airimu_output['gyro_cov']                # (128, 200, 3)
```

---

## 2. AirIO Network (Velocity Prediction)

### Input Features

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **Corrected Acc** | `(batch, 200, 3)` | m/s² | From AirIMU output |
| **Corrected Gyro** | `(batch, 200, 3)` | rad/s | From AirIMU output |
| **Orientation** | `(batch, 200, 4)` | quaternion | From AirIMU output [qw, qx, qy, qz] |

**Total Input Shape**: `(batch_size, window_size, 10)`
- `10`: 3 acc + 3 gyro + 4 quaternion

**Note**: If using `codewithrot` (with attitude encoding), orientation is encoded into rotation matrix features.

### Output Features

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **Velocity** | `(batch, 200, 3)` | m/s | Predicted velocity in body frame [vx, vy, vz] |
| **Covariance** | `(batch, 200, 3, 3)` | m²/s² | Velocity uncertainty (optional) |

### Example Code

```python
# Input (from AirIMU)
corrected_acc = torch.randn(128, 200, 3)
corrected_gyro = torch.randn(128, 200, 3)
orientation = torch.randn(128, 200, 4)  # quaternions

# Forward pass
airio_output = airio_model(corrected_acc, corrected_gyro, orientation)

# Output
velocity = airio_output['velocity']     # (128, 200, 3)
cov = airio_output.get('cov', None)     # (128, 200, 3, 3) or None
```

---

## 3. EKF (Extended Kalman Filter)

### State Vector

| Component | Size | Unit | Description |
|-----------|------|------|-------------|
| **Rotation** | 3 | - | SO(3) log representation [rx, ry, rz] |
| **Velocity** | 3 | m/s | Global frame velocity [vx, vy, vz] |
| **Position** | 3 | m | Global frame position [px, py, pz] |
| **Gyro Bias** | 3 | rad/s | Gyroscope bias [bgx, bgy, bgz] |
| **Acc Bias** | 3 | m/s² | Accelerometer bias [bax, bay, baz] |

**Total State Size**: 15

### Input (per timestep)

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **Gyro** | `(3,)` | rad/s | Current gyroscope reading |
| **Acc** | `(3,)` | m/s² | Current accelerometer reading |
| **dt** | `(1,)` | s | Time interval |
| **Velocity Obs** | `(3,)` | m/s | From AirIO (when available) |

### Output (per timestep)

| Feature | Shape | Unit | Description |
|---------|-------|------|-------------|
| **State** | `(15,)` | mixed | Full state vector |
| **Covariance** | `(15, 15)` | mixed | State uncertainty |

### Example Code

```python
# State
state = torch.zeros(15)  # [R(3), V(3), P(3), bg(3), ba(3)]
P = torch.eye(15) * 0.1  # Covariance

# Input
gyro = torch.tensor([0.01, -0.02, 1.25])
acc = torch.tensor([0.05, -0.03, 9.81])
dt = torch.tensor(0.005)
velocity_obs = torch.tensor([1.0, 0.5, 0.0])  # From AirIO

# EKF update
input_ekf = torch.cat([gyro, acc, state[9:12], state[12:15]])
state, P = ekf(state=state, obs=velocity_obs, input=input_ekf, P=P, dt=dt)

# Extract results
rotation = state[0:3]      # SO(3) log
velocity = state[3:6]      # m/s
position = state[6:9]      # m
gyro_bias = state[9:12]    # rad/s
acc_bias = state[12:15]    # m/s²
```

---

## 4. Complete Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Raw IMU Sensor                            │
│  Output: acc (3,), gyro (3,), dt (1,)                       │
│  Rate: 200 Hz                                                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                   [Buffer 200 samples]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    AirIMU Network                            │
│  Input:  acc (batch, 200, 3), gyro (batch, 200, 3)         │
│  Output: corrected_acc (batch, 200, 3)                      │
│          corrected_gyro (batch, 200, 3)                     │
│          orientation (batch, 200, 4)                        │
│  Rate: 1 Hz (every 200 samples)                             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    AirIO Network                             │
│  Input:  corrected_acc (batch, 200, 3)                      │
│          corrected_gyro (batch, 200, 3)                     │
│          orientation (batch, 200, 4)                        │
│  Output: velocity (batch, 200, 3)                           │
│  Rate: 1 Hz (every 200 samples)                             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                  [Extract latest velocity]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    EKF Fusion                                │
│  Input:  gyro (3,), acc (3,), dt (1,)                       │
│          velocity_obs (3,) [when available]                 │
│  State:  [R(3), V(3), P(3), bg(3), ba(3)] = 15D            │
│  Output: position (3,), velocity (3,), orientation (SO3)    │
│  Rate: 200 Hz (every sample)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Network Architectures

### AirIMU Network (Typical)

```
Input: (batch, 200, 6)
    ↓
[CNN Encoder]
    Conv1D(6 → 64, kernel=5)
    Conv1D(64 → 128, kernel=5)
    Conv1D(128 → 256, kernel=5)
    ↓
[GRU Layers]
    GRU(256 → 256, 2 layers)
    ↓
[Output Heads]
    Linear(256 → 3)  # Corrected Acc
    Linear(256 → 3)  # Corrected Gyro
    Linear(256 → 4)  # Orientation
    Linear(256 → 3)  # Acc Cov
    Linear(256 → 3)  # Gyro Cov
    ↓
Output: Multiple tensors (batch, 200, *)
```

### AirIO Network (codewithrot)

```
Input: (batch, 200, 10)  # acc(3) + gyro(3) + quat(4)
    ↓
[Attitude Encoding]
    Quaternion → Rotation Matrix (3x3)
    Flatten → 9 features
    ↓
Combined: (batch, 200, 15)  # acc(3) + gyro(3) + rot_matrix(9)
    ↓
[CNN Encoder]
    Conv1D(15 → 64, kernel=5)
    Conv1D(64 → 128, kernel=5)
    Conv1D(128 → 256, kernel=5)
    ↓
[GRU Layers]
    GRU(256 → 256, 2 layers)
    ↓
[Output Head]
    Linear(256 → 3)  # Velocity
    ↓
Output: (batch, 200, 3)
```

---

## 6. Practical Example: Single Sample Processing

```python
import torch
import pypose as pp
from collections import deque

# Initialize
window_size = 200
imu_buffer = deque(maxlen=window_size)
state = torch.zeros(15)
P = torch.eye(15) * 0.1

# Read IMU sample
imu_sample = {
    'acc': torch.tensor([0.05, -0.03, 9.81]),    # m/s²
    'gyro': torch.tensor([0.01, -0.02, 1.25]),   # rad/s
    'dt': 0.005                                   # seconds
}

# Add to buffer
imu_buffer.append(imu_sample)

# When buffer is full (200 samples)
if len(imu_buffer) == window_size:
    # Prepare sequences
    acc_seq = torch.stack([s['acc'] for s in imu_buffer]).unsqueeze(0)    # (1, 200, 3)
    gyro_seq = torch.stack([s['gyro'] for s in imu_buffer]).unsqueeze(0)  # (1, 200, 3)
    
    # AirIMU
    airimu_out = airimu_model(acc_seq, gyro_seq)
    corrected_acc = airimu_out['corrected_acc']    # (1, 200, 3)
    corrected_gyro = airimu_out['corrected_gyro']  # (1, 200, 3)
    orientation = airimu_out['rotation']           # (1, 200, 4)
    
    # AirIO
    airio_out = airio_model(corrected_acc, corrected_gyro, orientation)
    velocity = airio_out['velocity']               # (1, 200, 3)
    
    # Extract latest velocity
    velocity_obs = velocity[0, -1, :]              # (3,)
else:
    velocity_obs = None

# EKF (runs every sample)
gyro_bias = state[9:12]
acc_bias = state[12:15]
input_ekf = torch.cat([imu_sample['gyro'], imu_sample['acc'], gyro_bias, acc_bias])

if velocity_obs is not None:
    # Propagate + Update
    state, P = ekf(state=state, obs=velocity_obs, input=input_ekf, P=P, dt=imu_sample['dt'])
else:
    # Propagate only
    state, P = ekf.state_propogate(state=state, input=input_ekf, P=P, dt=imu_sample['dt'])

# Extract odometry
rotation = pp.so3(state[0:3]).Exp()  # SO3 object
velocity = state[3:6]                 # (3,) m/s
position = state[6:9]                 # (3,) m

print(f"Position: {position.numpy()}")
print(f"Velocity: {velocity.numpy()}")
```

---

## 7. Summary Table

| Component | Input Shape | Output Shape | Rate |
|-----------|-------------|--------------|------|
| **AirIMU** | (batch, 200, 6) | (batch, 200, 3+3+4+3+3) | 1 Hz |
| **AirIO** | (batch, 200, 10) | (batch, 200, 3) | 1 Hz |
| **EKF** | (3+3+1) per sample | (15,) state | 200 Hz |

### Memory Requirements

| Component | Parameters | Memory |
|-----------|-----------|--------|
| **AirIMU** | ~2-5M | ~20-50 MB |
| **AirIO** | ~1-3M | ~10-30 MB |
| **EKF** | 0 (analytical) | ~1 MB |
| **Buffer** | 200 samples × 6 | ~5 KB |

### Computational Cost

| Component | FLOPs | Latency (CPU) | Latency (GPU) |
|-----------|-------|---------------|---------------|
| **AirIMU** | ~50M | ~10-20 ms | ~2-5 ms |
| **AirIO** | ~30M | ~5-10 ms | ~1-3 ms |
| **EKF** | ~10K | ~0.1 ms | ~0.1 ms |

---

## 8. Key Takeaways

1. **Window-based processing**: Networks process 200 samples (1 second) at once
2. **Body frame**: All IMU data is in body frame
3. **Quaternions**: Orientation uses [w, x, y, z] format
4. **EKF state**: 15D vector with rotation, velocity, position, and biases
5. **Real-time capable**: Total latency ~15-30 ms, well below 5 ms requirement @ 200 Hz
