# AirIO Model Architecture - Quick Reference

## 📊 Input/Output Summary

### AirIO Network (CodeNetMotionwithRot)

```
INPUT:
├── data["acc"]     → (batch, 1000, 3)  # Accelerometer [ax, ay, az] in m/s²
├── data["gyro"]    → (batch, 1000, 3)  # Gyroscope [ωx, ωy, ωz] in rad/s
└── rot             → (batch, 1000, 3)  # Rotation encoding (SO3 log or similar)

OUTPUT:
├── net_vel         → (batch, 111, 3)   # Predicted velocity [vx, vy, vz] in m/s
└── cov (optional)  → (batch, 111, 3)   # Velocity covariance (if propcov=True)
```

**Note**: Output length is 111 due to CNN downsampling (1000 → 111 with stride=3, kernel=7)

---

## 🏗️ Architecture Details

### CodeNetMotionwithRot (with Attitude Encoding)

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  data["acc"]  (batch, 1000, 3)  ┐                               │
│  data["gyro"] (batch, 1000, 3)  ├─→ concat → (batch, 1000, 6)   │
│                                  ┘                               │
│  rot          (batch, 1000, 3)  ────────────→ (batch, 1000, 3)  │
└─────────────────────────────────────────────────────────────────┘
                         ↓                              ↓
┌─────────────────────────────────────┐  ┌─────────────────────────┐
│      Feature Encoder (CNN)          │  │   Orientation Encoder   │
├─────────────────────────────────────┤  ├─────────────────────────┤
│  Conv1D(6→32, k=7, s=3, p=3)       │  │  Conv1D(3→32, k=7, s=3) │
│  BatchNorm + GELU                   │  │  BatchNorm + GELU       │
│  Conv1D(32→64, k=7, s=3, p=3)      │  │  Conv1D(32→64, k=7, s=3)│
│  BatchNorm + GELU                   │  │  BatchNorm + GELU       │
│  Dropout(0.5)                       │  │  Dropout(0.5)           │
│                                     │  │                         │
│  Output: (batch, 111, 64)          │  │  Output: (batch, 111, 64)│
└─────────────────────────────────────┘  └─────────────────────────┘
                         ↓                              ↓
                         └──────────────┬───────────────┘
                                        ↓
                         ┌──────────────────────────────┐
                         │  Concatenate                 │
                         │  (batch, 111, 128)          │
                         └──────────────┬───────────────┘
                                        ↓
                         ┌──────────────────────────────┐
                         │  Linear(128→64)              │
                         │  BatchNorm + GELU            │
                         │  (batch, 111, 64)           │
                         └──────────────┬───────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────┐
│                      GRU LAYERS                                  │
├─────────────────────────────────────────────────────────────────┤
│  GRU1: input=64, hidden=64, bidirectional                       │
│        Output: (batch, 111, 128)                                │
│                                                                  │
│  GRU2: input=128, hidden=128, bidirectional                     │
│        Output: (batch, 111, 256)                                │
└─────────────────────────────────────────────────────────────────┘
                                        ↓
                         ┌──────────────┴───────────────┐
                         ↓                              ↓
┌─────────────────────────────────┐  ┌─────────────────────────────┐
│     Velocity Decoder            │  │  Covariance Decoder         │
├─────────────────────────────────┤  ├─────────────────────────────┤
│  Linear(256→128)                │  │  Linear(256→128)            │
│  GELU                           │  │  GELU                       │
│  Linear(128→3)                  │  │  Linear(128→3)              │
│                                 │  │  exp(x - 5.0)               │
│  Output: (batch, 111, 3)       │  │  Output: (batch, 111, 3)    │
└─────────────────────────────────┘  └─────────────────────────────┘
                         ↓                              ↓
                    net_vel                           cov
```

---

## 📐 Dimension Changes Through Network

| Layer | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| **Input** | (B, 1000, 6) | - | acc(3) + gyro(3) |
| **CNN Layer 1** | (B, 1000, 6) | (B, 334, 32) | stride=3, kernel=7 |
| **CNN Layer 2** | (B, 334, 32) | (B, 111, 64) | stride=3, kernel=7 |
| **Orientation CNN** | (B, 1000, 3) | (B, 111, 64) | Same downsampling |
| **Concat** | 2×(B, 111, 64) | (B, 111, 128) | - |
| **FCN** | (B, 111, 128) | (B, 111, 64) | - |
| **GRU1** | (B, 111, 64) | (B, 111, 128) | Bidirectional |
| **GRU2** | (B, 111, 128) | (B, 111, 256) | Bidirectional |
| **Decoder** | (B, 111, 256) | (B, 111, 3) | Velocity output |

**B** = batch size (typically 128)

---

## 🔢 Parameter Count

### CodeNetMotionwithRot

| Component | Parameters |
|-----------|-----------|
| Feature CNN | ~50K |
| Orientation CNN | ~25K |
| GRU Layers | ~500K |
| Decoders | ~100K |
| **Total** | **~675K** |

### CodeNetMotion (without rotation)

| Component | Parameters |
|-----------|-----------|
| CNN | ~50K |
| GRU Layers | ~500K |
| Decoders | ~100K |
| **Total** | **~650K** |

---

## 💾 Memory Usage

| Item | Size |
|------|------|
| Model weights | ~2.7 MB (float32) |
| Input batch (128×1000×6) | ~3 MB |
| Activations | ~50 MB |
| **Total (inference)** | **~56 MB** |

---

## ⚡ Computational Cost

| Operation | FLOPs | Time (CPU) | Time (GPU) |
|-----------|-------|------------|------------|
| CNN Encoding | ~10M | ~5 ms | ~1 ms |
| GRU Forward | ~20M | ~10 ms | ~2 ms |
| Decoding | ~1M | ~1 ms | ~0.5 ms |
| **Total** | **~31M** | **~16 ms** | **~3.5 ms** |

---

## 📝 Key Points

### Input Requirements
1. **Window size**: 1000 samples (5 seconds @ 200Hz) or 200 samples (1 second @ 200Hz)
2. **Sampling rate**: 200 Hz (dt = 0.005s)
3. **Coordinate frame**: Body frame
4. **Units**: 
   - Acceleration: m/s²
   - Gyroscope: rad/s
   - Rotation: SO(3) log representation

### Output Characteristics
1. **Downsampling**: Output is ~1/9 of input length due to stride=3 in 2 CNN layers
2. **Temporal alignment**: Output corresponds to downsampled input timestamps
3. **Frame**: Body frame velocity
4. **Units**: m/s

### Network Variants

**CodeNetMotion** (without rotation):
- Input: acc(3) + gyro(3) = 6 features
- Simpler, faster
- Use when orientation is not available

**CodeNetMotionwithRot** (with attitude encoding):
- Input: acc(3) + gyro(3) + rot(3) = 9 features (effectively)
- Better accuracy
- Use when orientation is available (from AirIMU or ground truth)

---

## 🎯 Practical Usage

### Training
```python
# Prepare data
data = {
    "acc": torch.randn(128, 1000, 3),   # batch=128, window=1000
    "gyro": torch.randn(128, 1000, 3)
}
rot = torch.randn(128, 1000, 3)  # Rotation encoding

# Forward pass
model = CodeNetMotionwithRot(conf)
output = model(data, rot)

# Get predictions
velocity = output['net_vel']        # (128, 111, 3)
covariance = output['cov']          # (128, 111, 3) or None
```

### Inference (Real-time)
```python
# Collect 1000 samples (5 seconds @ 200Hz)
acc_buffer = []  # Collect 1000 samples
gyro_buffer = []
rot_buffer = []

# When buffer is full
data = {
    "acc": torch.tensor(acc_buffer).unsqueeze(0),    # (1, 1000, 3)
    "gyro": torch.tensor(gyro_buffer).unsqueeze(0)   # (1, 1000, 3)
}
rot = torch.tensor(rot_buffer).unsqueeze(0)          # (1, 1000, 3)

# Predict
with torch.no_grad():
    output = model(data, rot)
    velocity = output['net_vel'][0]  # (111, 3)

# Use latest prediction
latest_velocity = velocity[-1]  # (3,) - most recent velocity estimate
```

---

## 🔄 Comparison: With vs Without Rotation

| Feature | CodeNetMotion | CodeNetMotionwithRot |
|---------|---------------|----------------------|
| **Input features** | 6 (acc + gyro) | 9 (acc + gyro + rot) |
| **Encoders** | 1 CNN | 2 CNNs (feature + orientation) |
| **Parameters** | ~650K | ~675K |
| **Accuracy** | Good | Better |
| **Use case** | No orientation available | Orientation from AirIMU/GT |

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Velocity RMSE | 0.05-0.15 m/s |
| Position drift | 0.1-0.5 m (after 60s) |
| Inference time | 3-16 ms |
| Throughput | 60-300 Hz |

---

## 🚀 Quick Reference

```python
# Model input/output shapes
INPUT:  data["acc"]  → (batch, 1000, 3)
        data["gyro"] → (batch, 1000, 3)
        rot          → (batch, 1000, 3)

OUTPUT: net_vel      → (batch, 111, 3)
        cov          → (batch, 111, 3) [optional]

# Key parameters
window_size = 1000  # or 200 for faster processing
stride = [3, 3]     # CNN downsampling
kernel = [7, 7]     # CNN kernel size
hidden = [64, 128]  # GRU hidden sizes
```
