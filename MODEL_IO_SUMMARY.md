# Model I/O - Ultra Quick Reference

## AirIO Network (CodeNetMotionwithRot)

### Input
```python
data = {
    "acc": (batch, 1000, 3),   # Accelerometer [ax, ay, az] in m/s²
    "gyro": (batch, 1000, 3)   # Gyroscope [ωx, ωy, ωz] in rad/s
}
rot = (batch, 1000, 3)         # Rotation encoding (SO3 log)
```

### Output
```python
{
    "net_vel": (batch, 111, 3),  # Velocity [vx, vy, vz] in m/s
    "cov": (batch, 111, 3)       # Covariance (optional)
}
```

### Key Facts
- **Window**: 1000 samples = 5 seconds @ 200Hz (or 200 samples = 1 second)
- **Downsampling**: 1000 → 111 (stride=3 in 2 CNN layers)
- **Frame**: Body frame
- **Parameters**: ~675K
- **Inference**: ~3-16 ms

---

## Real-Time Usage

```python
# 1. Collect window of data
acc_window = torch.tensor(acc_buffer)    # (1000, 3)
gyro_window = torch.tensor(gyro_buffer)  # (1000, 3)
rot_window = torch.tensor(rot_buffer)    # (1000, 3)

# 2. Add batch dimension
data = {
    "acc": acc_window.unsqueeze(0),      # (1, 1000, 3)
    "gyro": gyro_window.unsqueeze(0)     # (1, 1000, 3)
}
rot = rot_window.unsqueeze(0)            # (1, 1000, 3)

# 3. Predict
output = model(data, rot)
velocity = output['net_vel'][0, -1, :]   # (3,) - latest velocity

# 4. Use in EKF
ekf.update(velocity_measurement=velocity)
```

---

## Complete Pipeline

```
IMU Sensor (200 Hz)
    ↓ acc(3), gyro(3), dt(1)
Buffer (1000 samples)
    ↓
AirIO Network (1 Hz)
    ↓ velocity(3)
EKF (200 Hz)
    ↓ position(3), velocity(3), orientation(SO3)
Output
```

---

## Files to Read

1. **MODEL_ARCHITECTURE_QUICK_REF.md** - Detailed architecture
2. **MODEL_IO_SPECIFICATIONS.md** - Complete specifications
3. **model/code.py** - Actual implementation
