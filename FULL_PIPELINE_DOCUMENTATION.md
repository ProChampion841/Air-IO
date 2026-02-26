# AirIO Full Pipeline Documentation: IMU-Only Odometry

## Overview
This document describes the complete pipeline for performing odometry estimation using only IMU sensors through the AirIO architecture.

---

## Architecture Components

### 1. **AirIMU Network** (IMU Correction)
- **Input**: Raw IMU measurements (accelerometer + gyroscope)
- **Output**: 
  - Corrected IMU measurements
  - Orientation estimates (rotation)
  - Measurement uncertainties (covariance)
- **Purpose**: Corrects systematic errors and biases in raw IMU data

### 2. **AirIO Network** (Velocity Estimation)
- **Input**: 
  - Corrected/Raw IMU data
  - Orientation (from AirIMU, integration, or ground truth)
- **Output**:
  - Body-frame or global-frame velocity
  - Velocity uncertainty (covariance)
- **Purpose**: Predicts velocity with enhanced feature observability using attitude encoding

### 3. **Extended Kalman Filter (EKF)** (State Fusion)
- **Input**:
  - AirIMU corrected measurements + uncertainties
  - AirIO velocity predictions + uncertainties
- **Output**:
  - Full state estimate: Rotation (R), Velocity (V), Position (P), Biases (bg, ba)
  - State covariance
- **Purpose**: Fuses predictions with uncertainties to produce optimal odometry estimate

---

## Complete Pipeline Flow

```
┌─────────────────┐
│   Raw IMU Data  │
│  (acc + gyro)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      AirIMU Network             │
│  ┌──────────────────────────┐   │
│  │ • Correct IMU biases     │   │
│  │ • Estimate orientation   │   │
│  │ • Compute uncertainties  │   │
│  └──────────────────────────┘   │
└────────┬────────────────────────┘
         │
         ├─────────────┬──────────────┐
         ▼             ▼              ▼
   Corrected IMU   Orientation   Uncertainties
         │             │              │
         └─────────────┴──────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │     AirIO Network           │
         │  ┌──────────────────────┐   │
         │  │ • Attitude encoding  │   │
         │  │ • Predict velocity   │   │
         │  │ • Estimate covariance│   │
         │  └──────────────────────┘   │
         └────────┬────────────────────┘
                  │
                  ▼
            Velocity + Cov
                  │
                  ▼
         ┌─────────────────────────────┐
         │   Extended Kalman Filter    │
         │  ┌──────────────────────┐   │
         │  │ State Propagation:   │   │
         │  │   R, V, P, bg, ba    │   │
         │  │                      │   │
         │  │ Measurement Update:  │   │
         │  │   Velocity obs       │   │
         │  └──────────────────────┘   │
         └────────┬────────────────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │  Final Odometry     │
         │  • Position (x,y,z) │
         │  • Orientation (R)  │
         │  • Velocity (v)     │
         └─────────────────────┘
```

---

## Step-by-Step Implementation

### **Phase 1: Data Preparation**

#### 1.1 Download Datasets
```bash
# EuRoC Dataset (simplified version)
wget https://github.com/Air-IO/Air-IO/releases/download/datasets/EuRoC-Dataset.zip

# Blackbird Dataset
wget https://github.com/Air-IO/Air-IO/releases/download/datasets/Blackbird.zip

# Pegasus Dataset
wget https://github.com/Air-IO/Air-IO/releases/download/datasets/PegasusDataset.zip
```

#### 1.2 Download Pre-trained Models
```bash
# AirIMU Models
wget https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_EuRoC.zip
wget https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_blackbird.zip
wget https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_pegasus.zip

# AirIO Models
wget https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_EuRoC.zip
wget https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Blackbird.zip
wget https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Pegasus.zip
```

---

### **Phase 2: AirIMU - IMU Correction**

#### 2.1 Train AirIMU (Optional - if using custom data)
```bash
# Visit https://airimu.github.io for AirIMU training instructions
```

#### 2.2 Generate Orientation from AirIMU
```bash
python3 evaluation/save_ori.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp /path/to/AirIMU_net_output.pickle
```

**Output**: `orientation_output.pickle` containing:
- `airimu_rot`: AirIMU-corrected orientation
- `inte_rot`: Raw IMU integrated orientation

---

### **Phase 3: AirIO - Velocity Estimation**

#### 3.1 Configure Dataset
Edit `configs/datasets/EuRoC/Euroc_body.conf`:
```yaml
train:
{
    mode: train
    coordinate: body_coord  # or glob_coord
    remove_g: False
    rot_type: airimu  # Options: airimu, integration, or None (GT)
    rot_path: /path/to/orientation_output.pickle
    data_list: [{
        name: EuRoC
        window_size: 1000
        step_size: 10
        data_root: /path/to/EuRoC-Dataset
        data_drive: [MH_01_easy, MH_02_easy]
    }]
    gravity: 9.81007
}
```

#### 3.2 Train AirIO Network
```bash
python3 train_motion.py \
    --config configs/EuRoC/motion_body_rot.conf \
    --log
```

**Training Config** (`configs/EuRoC/motion_body_rot.conf`):
```yaml
dataset: {
    include "../datasets/EuRoC/Euroc_body.conf"
    collate: {type: motion}
}

train: {
    network: codewithrot  # With attitude encoding
    lr: 1e-3
    min_lr: 1e-5
    batch_size: 128
    max_epoches: 100
    patience: 5
    factor: 0.2
    weight_decay: 1e-4
}
```

#### 3.3 Run AirIO Inference
```bash
python3 inference_motion.py \
    --config configs/EuRoC/motion_body_rot.conf
```

**Output**: `experiments/euroc/motion_body_rot/net_output.pickle` containing:
- `net_vel`: Predicted velocity
- `cov`: Velocity covariance
- `ts`: Timestamps

#### 3.4 Evaluate AirIO Results
```bash
python3 evaluation/evaluate_motion.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp experiments/euroc/motion_body_rot \
    --seqlen 500
```

---

### **Phase 4: EKF Fusion**

#### 4.1 Standard EKF
```bash
python3 EKF/IMUofflinerunner.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp experiments/euroc/motion_body_rot \
    --airimu_exp /path/to/AirIMU_results \
    --savedir ./EKFresult/euroc_results
```

**EKF State Vector** (15-dimensional):
```
[R (3), V (3), P (3), bg (3), ba (3)]
```
- R: Rotation (SO3 log representation)
- V: Velocity
- P: Position
- bg: Gyroscope bias
- ba: Accelerometer bias

**EKF Process**:
1. **Propagation**: Use corrected IMU to propagate state
2. **Update**: Use AirIO velocity as measurement
3. **Uncertainty**: Combine AirIMU and AirIO covariances

#### 4.2 CasADi-based EKF (Faster)
```bash
python3 EKF/casADI_EKF/casADI_EKFrunner.py \
    --conf configs/casADI_EKF/EuRoC/euroc.yaml
```

**Output**: 
- `{sequence}_ekf_poses.npy`: Trajectory (position + orientation)
- `{sequence}_ekf_result.npy`: Full state history + covariances

#### 4.3 Evaluate EKF Results
```bash
python3 evaluation/evaluate_ekf.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp EKFresult/euroc_results \
    --seqlen 1000
```

---

## Network Architectures

### AirIO Network Options

#### 1. **CodeNetMotionwithRot** (Recommended)
- Uses attitude encoding (rotation matrix features)
- Better observability of IMU features
- Config: `network: codewithrot`

#### 2. **CodeNetMotion** (Baseline)
- Standard architecture without attitude encoding
- Config: `network: codenetmotion`

### Coordinate Frame Options

#### 1. **Body Frame** (`coordinate: body_coord`)
- IMU measurements in body frame
- Predicts body-frame velocity
- Requires rotation for global position

#### 2. **Global Frame** (`coordinate: glob_coord`)
- IMU rotated to global frame
- Predicts global-frame velocity
- Direct integration to position

#### 3. **Global Frame without Gravity** (`remove_g: True`)
- Removes gravity component from accelerometer
- Used with global frame representation

---

## Key Configuration Parameters

### Dataset Configuration
```yaml
window_size: 1000    # Sequence length (frames)
step_size: 10        # Sliding window step
rot_type: airimu     # Rotation source: airimu/integration/None
coordinate: body_coord  # Frame: body_coord/glob_coord
remove_g: False      # Remove gravity: True/False
```

### Training Configuration
```yaml
lr: 1e-3            # Learning rate
batch_size: 128     # Batch size
max_epoches: 100    # Maximum epochs
patience: 5         # LR scheduler patience
weight_decay: 1e-4  # Optimizer weight decay
```

### EKF Configuration
```python
Q = torch.eye(12) * q**2  # Process noise (IMU + bias)
R = torch.eye(3) * r**2   # Measurement noise (velocity)
P = torch.eye(15) * p**2  # Initial state covariance
```

---

## Output Data Structures

### AirIMU Output (`net_output.pickle`)
```python
{
    'sequence_name': {
        'correction_acc': torch.Tensor,      # Accelerometer correction
        'correction_gyro': torch.Tensor,     # Gyroscope correction
        'corrected_acc': torch.Tensor,       # Corrected accelerometer
        'corrected_gyro': torch.Tensor,      # Corrected gyroscope
        'acc_cov': torch.Tensor,             # Accelerometer covariance
        'gyro_cov': torch.Tensor,            # Gyroscope covariance
        'rot': pp.SO3Type,                   # Orientation estimate
        'dt': torch.Tensor                   # Time intervals
    }
}
```

### AirIO Output (`net_output.pickle`)
```python
{
    'sequence_name': {
        'net_vel': torch.Tensor,  # Predicted velocity [N, 3]
        'cov': torch.Tensor,      # Velocity covariance [N, 3, 3]
        'ts': torch.Tensor        # Timestamps [N]
    }
}
```

### EKF Output (`{seq}_ekf_result.npy`)
```python
{
    'state': torch.Tensor,     # State history [N, 15]
    'covariance': torch.Tensor # Covariance history [N, 15, 15]
}
# State: [R(3), V(3), P(3), bg(3), ba(3)]
```

### EKF Poses (`{seq}_ekf_poses.npy`)
```python
{
    'position': np.ndarray,    # [N, 3] - xyz position
    'rotation': np.ndarray,    # [N, 4] - quaternion (w,x,y,z)
    'timestamp': np.ndarray    # [N] - timestamps
}
```

---

## Evaluation Metrics

### Trajectory Metrics
- **ATE (Absolute Trajectory Error)**: Overall position accuracy
- **RTE (Relative Trajectory Error)**: Local consistency over segments
- **Orientation Error**: Rotation accuracy in degrees

### Velocity Metrics
- **Velocity RMSE**: Root mean square error of velocity prediction
- **Velocity MAE**: Mean absolute error

---

## Quick Start Example (EuRoC Dataset)

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download data and models
# (Download EuRoC dataset and pre-trained models as shown above)

# 3. Set data paths in config
# Edit configs/datasets/EuRoC/Euroc_body.conf
# Set data_root to your dataset location

# 4. Run inference with pre-trained AirIO
python3 inference_motion.py \
    --config configs/EuRoC/motion_body_rot.conf

# 5. Run EKF with pre-trained models
python3 EKF/IMUofflinerunner.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp experiments/euroc/motion_body_rot \
    --airimu_exp AirIMU_Model_Results/AirIMU_EuRoC \
    --savedir ./EKFresult/euroc

# 6. Evaluate results
python3 evaluation/evaluate_ekf.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp EKFresult/euroc \
    --seqlen 1000
```

---

## Training from Scratch

### Complete Training Pipeline

```bash
# Step 1: Train AirIMU (see https://airimu.github.io)
# Produces: AirIMU_net_output.pickle

# Step 2: Generate orientation file
python3 evaluation/save_ori.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp /path/to/AirIMU_net_output.pickle

# Step 3: Update config with orientation path
# Edit configs/datasets/EuRoC/Euroc_body.conf:
#   rot_type: airimu
#   rot_path: /path/to/orientation_output.pickle

# Step 4: Train AirIO
python3 train_motion.py \
    --config configs/EuRoC/motion_body_rot.conf

# Step 5: Run inference
python3 inference_motion.py \
    --config configs/EuRoC/motion_body_rot.conf

# Step 6: Run EKF
python3 EKF/IMUofflinerunner.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp experiments/euroc/motion_body_rot \
    --airimu_exp /path/to/AirIMU_results

# Step 7: Evaluate
python3 evaluation/evaluate_ekf.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp EKFresult/euroc \
    --seqlen 1000
```

---

## Custom Dataset Integration

### 1. Create Dataset Class
File: `datasets/CustomDataset.py`
```python
class CustomDataset(BaseDataset):
    def __init__(self, data_root, data_name, ...):
        # Load IMU data
        self.data = {
            "dt": time_intervals,
            "time": timestamps,
            "acc": accelerometer,  # [N, 3]
            "gyro": gyroscope,     # [N, 3]
            "gt_orientation": pp.SO3(...),
            "gt_translation": position,  # [N, 3]
            "velocity": velocity   # [N, 3]
        }
        
        # Apply transformations
        self.set_orientation(rot_path, data_name, rot_type)
        self.update_coordinate(coordinate, mode)
        self.remove_gravity(remove_g)
```

### 2. Create Dataset Config
File: `configs/datasets/Custom/custom_body.conf`
```yaml
train: {
    mode: train
    coordinate: body_coord
    remove_g: False
    data_list: [{
        name: CustomDataset  # Must match class name
        window_size: 1000
        step_size: 10
        data_root: /path/to/custom/data
        data_drive: [seq1, seq2, seq3]
    }]
    gravity: 9.81
}
```

### 3. Create Training Config
File: `configs/Custom/motion_body_rot.conf`
```yaml
dataset: {
    include "../datasets/Custom/custom_body.conf"
    collate: {type: motion}
}

train: {
    network: codewithrot
    lr: 1e-3
    batch_size: 128
    max_epoches: 100
}
```

### 4. Train and Evaluate
```bash
python3 train_motion.py --config configs/Custom/motion_body_rot.conf
python3 inference_motion.py --config configs/Custom/motion_body_rot.conf
python3 EKF/IMUofflinerunner.py --dataconf configs/datasets/Custom/custom_body.conf --exp experiments/custom/motion_body_rot
```

---

## Troubleshooting

### Common Issues

**1. Orientation file not found**
- Ensure `rot_path` in dataset config points to valid `orientation_output.pickle`
- Generate using `evaluation/save_ori.py` with AirIMU results

**2. Poor EKF performance**
- Tune Q, R, P matrices in `EKF/IMUofflinerunner.py`
- Check AirIO velocity predictions quality first
- Verify AirIMU corrections are reasonable

**3. Training divergence**
- Reduce learning rate
- Check data normalization
- Verify ground truth quality

**4. Memory issues**
- Reduce `window_size` in dataset config
- Reduce `batch_size` in training config
- Use gradient accumulation

---

## Performance Tips

1. **Use CasADi EKF** for faster Jacobian computation
2. **Pre-compute orientations** before training AirIO
3. **Use body-frame representation** for better generalization
4. **Enable attitude encoding** (`codewithrot`) for better accuracy
5. **Tune EKF covariances** based on sensor characteristics

---

## Citation

```bibtex
@misc{qiu2025airiolearninginertialodometry,
    title={AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability}, 
    author={Yuheng Qiu and Can Xu and Yutian Chen and Shibo Zhao and Junyi Geng and Sebastian Scherer},
    year={2025},
    eprint={2501.15659},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2501.15659}
}
```

---

## Additional Resources

- **Project Homepage**: https://air-io.github.io/
- **AirIMU Project**: https://airimu.github.io/
- **Paper**: http://arxiv.org/abs/2501.15659
- **PyPose Documentation**: https://github.com/pypose/pypose
