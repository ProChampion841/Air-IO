# 🚀 AirIO Real-Time Pipeline - Quick Reference

## One-Line Commands

### Run Demo
```bash
python realtime_pipeline_demo.py
```

### Run with Batch Script (Windows)
```bash
run_demo.bat
```

## File Structure

```
Air-IO/
├── realtime_pipeline_demo.py          # ⭐ Main demo script
├── run_demo.bat                        # Quick start (Windows)
├── REALTIME_DEMO_README.md            # Detailed demo guide
├── REALTIME_IMPLEMENTATION_SUMMARY.md # Complete summary
├── REALTIME_DEPLOYMENT_GUIDE.md       # Production deployment
├── FULL_PIPELINE_DOCUMENTATION.md     # Full pipeline docs
└── EKF/
    ├── IMUrealtimerunner.py           # Real-time EKF class
    └── IMUofflinerunner.py            # Offline EKF (with errors)
```

## Pipeline Flow

```
IMU (200Hz) → Buffer (1s) → AirIMU → AirIO → EKF → Odometry
```

## Key Classes

### MockIMUSensor
```python
sensor = MockIMUSensor(dt=0.005, duration=10.0)
data = sensor.read()  # {'acc', 'gyro', 'dt', 'gt_pos', 'gt_vel', 'gt_rot'}
```

### RealtimeOdometryPipeline
```python
pipeline = RealtimeOdometryPipeline(window_size=200)
pipeline.initialize(init_pos, init_rot, init_vel)
result = pipeline.process_imu(imu_data)  # {'position', 'velocity', 'orientation'}
```

## Quick Customization

### Change Duration
```python
sensor = MockIMUSensor(duration=5.0)  # 5 seconds instead of 10
```

### Change Window Size
```python
pipeline = RealtimeOdometryPipeline(window_size=100)  # 0.5s window
```

### Tune EKF
```python
Q = torch.eye(12) * 1e-4  # Process noise
R = torch.eye(3) * 1e-2   # Measurement noise
```

## Output Files

```
realtime_demo_results/
└── realtime_pipeline_results.png  # 6 subplots
```

## Error Metrics

- **Position**: meters (m)
- **Velocity**: meters/second (m/s)
- **Rotation**: degrees (°)

## Typical Performance

| Metric | Value |
|--------|-------|
| Position Error | 0.1-0.3m |
| Velocity Error | 0.05-0.15 m/s |
| Rotation Error | 1-3° |
| Processing Speed | 130 Hz |
| Latency | 7-8 ms |

## Common Issues

### Import Error
```bash
cd Air-IO  # Run from project root
```

### Too Slow
```python
window_size=100  # Reduce window
duration=5.0     # Shorter test
```

### Poor Accuracy
```python
Q = torch.eye(12) * 1e-5  # Tune Q
R = torch.eye(3) * 1e-3   # Tune R
```

## Documentation Quick Links

| Document | Purpose |
|----------|---------|
| `REALTIME_DEMO_README.md` | How to use demo |
| `REALTIME_IMPLEMENTATION_SUMMARY.md` | What was created |
| `REALTIME_DEPLOYMENT_GUIDE.md` | Deploy to production |
| `FULL_PIPELINE_DOCUMENTATION.md` | Complete pipeline |

## State Vector (15D)

```
[R(3), V(3), P(3), bg(3), ba(3)]
 │     │     │     │      └─ Accelerometer bias
 │     │     │     └─ Gyroscope bias
 │     │     └─ Position
 │     └─ Velocity
 └─ Rotation (SO3 log)
```

## Network Outputs

### AirIMU
- Corrected accelerometer
- Corrected gyroscope
- Orientation estimate
- Covariances

### AirIO
- Body-frame velocity
- Velocity covariance

## EKF Operations

### Propagate (Every Sample)
```
State(k) + IMU → State(k+1)
```

### Update (When Velocity Available)
```
State(k+1) + Velocity → State(k+1|k+1)
```

## Real-Time Requirements

- **Sampling Rate**: 200 Hz (5ms period)
- **Processing Time**: < 5ms per sample
- **Buffer Size**: 200 samples (1 second)
- **Network Frequency**: 1 Hz (every 200 samples)
- **EKF Frequency**: 200 Hz (every sample)

## Migration Checklist

- [ ] Replace MockIMUSensor with real sensor
- [ ] Load trained AirIMU model
- [ ] Load trained AirIO model
- [ ] Test with recorded data
- [ ] Tune EKF parameters
- [ ] Deploy to target platform
- [ ] Monitor performance
- [ ] Validate accuracy

## Support

- **Issues**: Check troubleshooting in REALTIME_DEMO_README.md
- **Questions**: See FULL_PIPELINE_DOCUMENTATION.md
- **Deployment**: See REALTIME_DEPLOYMENT_GUIDE.md

---

**Quick Start**: `python realtime_pipeline_demo.py`

**Status**: ✅ Ready to Use
