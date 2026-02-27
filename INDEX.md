# 📚 AirIO Complete Pipeline Documentation Index

## 🚀 Start Here

**New to AirIO?** → Read [QUICKSTART.md](QUICKSTART.md) (5 minutes)

**Want full details?** → Read [FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md) (30 minutes)

**Ready to deploy?** → See [Deployment Section](#deployment)

---

## 📖 Documentation Files

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** ⭐ START HERE
   - 5-step workflow from dataset to deployment
   - Essential commands
   - Arduino code example
   - Quick troubleshooting

2. **[PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md)**
   - Overview of complete pipeline
   - What was created and why
   - Performance expectations
   - Quick reference

### Detailed Guides
3. **[FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md)**
   - Complete walkthrough
   - Dataset preparation
   - Configuration files
   - Training process
   - Real-time deployment
   - Sensor setup
   - Troubleshooting

4. **[REALTIME_WITH_PRETRAINED_MODELS.md](REALTIME_WITH_PRETRAINED_MODELS.md)**
   - Using pre-trained models
   - Model loading
   - Network architecture notes

### Original Documentation
5. **[README.md](README.md)**
   - Original AirIO paper documentation
   - Dataset downloads
   - Pre-trained models
   - Citation

---

## 🛠️ Code Files

### Core Pipeline
- **`train_motion.py`** - Train AirIO network
- **`inference_motion.py`** - Run inference on test data
- **`evaluation/evaluate_motion.py`** - Evaluate and visualize results

### Real-Time Deployment
- **`realtime_sensor_interface.py`** ⭐ IMU sensor interface
  - `RealIMUSensor` class for serial communication
  - `IMUDataLogger` for logging raw data
  - `test_sensor_connection()` for testing

- **`realtime_deployment.py`** ⭐ Production deployment
  - `run_realtime_with_sensor()` for real-time processing
  - `visualize_results()` for plotting
  - Command-line interface

- **`realtime_pipeline_demo.py`** - Demo with mock/real models
  - `RealtimeOdometryPipeline` class
  - Mock networks for testing
  - Real network loading

### Dataset Templates
- **`datasets/CustomDataset_template.py`** ⭐ Template for your dataset
  - Copy and modify for your data format
  - Includes interpolation and velocity computation
  - Built-in testing

---

## 📋 Quick Command Reference

### Training
```bash
# Train AirIO
python train_motion.py --config configs/MyDataset/motion_body_rot.conf

# Run inference
python inference_motion.py --config configs/MyDataset/motion_body_rot.conf

# Evaluate
python evaluation/evaluate_motion.py \
    --dataconf configs/datasets/MyDataset/mydataset.conf \
    --exp experiments/mydataset/motion_body_rot \
    --seqlen 500
```

### Real-Time Deployment
```bash
# Test sensor
python realtime_sensor_interface.py --port /dev/ttyUSB0 --duration 5

# Run real-time odometry
python realtime_deployment.py \
    --port /dev/ttyUSB0 \
    --airio-model experiments/mydataset/motion_body_rot/best_model.pth \
    --duration 60 \
    --save \
    --log-imu

# Visualize results
python realtime_deployment.py --visualize realtime_results_*.npy
```

---

## 🎯 Workflow Overview

```
┌──────────────┐
│   Dataset    │  → Prepare CSV files (imu_data.csv, groundTruthPoses.csv)
└──────┬───────┘
       ↓
┌──────────────┐
│   Config     │  → Create dataset class and config files
└──────┬───────┘
       ↓
┌──────────────┐
│   Training   │  → python train_motion.py --config ...
└──────┬───────┘    Output: best_model.pth
       ↓
┌──────────────┐
│ Sensor Setup │  → Upload Arduino code, test connection
└──────┬───────┘
       ↓
┌──────────────┐
│  Deployment  │  → python realtime_deployment.py ...
└──────┬───────┘    Real-time odometry @ 200Hz
       ↓
┌──────────────┐
│Visualization │  → python realtime_deployment.py --visualize ...
└──────────────┘    3D trajectory, velocity plots
```

---

## 📁 File Structure

```
Air-IO/
├── 📚 Documentation
│   ├── QUICKSTART.md ⭐
│   ├── FULL_PIPELINE_GUIDE.md
│   ├── PIPELINE_SUMMARY.md
│   ├── REALTIME_WITH_PRETRAINED_MODELS.md
│   ├── INDEX.md (this file)
│   └── README.md (original)
│
├── 🔧 Real-Time Code
│   ├── realtime_sensor_interface.py ⭐
│   ├── realtime_deployment.py ⭐
│   └── realtime_pipeline_demo.py
│
├── 🎓 Training Code
│   ├── train_motion.py
│   ├── inference_motion.py
│   └── evaluation/
│
├── 📊 Datasets
│   ├── datasets/
│   │   ├── CustomDataset_template.py ⭐
│   │   ├── EuRoCdataset.py
│   │   └── ...
│   └── data/
│       └── YourDataset/
│
├── ⚙️ Configuration
│   └── configs/
│       ├── datasets/
│       └── YourDataset/
│
└── 🤖 Models
    ├── model/
    └── experiments/
```

---

## 🎓 Learning Path

### Beginner (Day 1)
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run demo: `python realtime_pipeline_demo.py`
3. Test sensor: `python realtime_sensor_interface.py --port /dev/ttyUSB0`

### Intermediate (Week 1)
1. Read [FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md)
2. Prepare your dataset using template
3. Create config files
4. Train your model

### Advanced (Week 2)
1. Deploy with real sensor
2. Tune EKF parameters
3. Optimize for your application
4. Integrate with your system

---

## 🔍 Find What You Need

### I want to...

**...understand the complete pipeline**
→ [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md)

**...train on my dataset**
→ [FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md) Phase 1-4

**...use pre-trained models**
→ [REALTIME_WITH_PRETRAINED_MODELS.md](REALTIME_WITH_PRETRAINED_MODELS.md)

**...deploy with real sensor**
→ [FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md) Phase 5

**...create custom dataset**
→ `datasets/CustomDataset_template.py`

**...test my IMU sensor**
→ `python realtime_sensor_interface.py --port /dev/ttyUSB0`

**...visualize results**
→ `python realtime_deployment.py --visualize results.npy`

**...troubleshoot issues**
→ [QUICKSTART.md](QUICKSTART.md) Troubleshooting section

---

## 💡 Key Concepts

### AirIO Pipeline
```
Raw IMU → [AirIMU] → Corrected IMU + Orientation
                              ↓
                         [AirIO] → Velocity Prediction
                              ↓
                          [EKF] → Full Odometry (R, V, P, biases)
```

### Data Flow
```
Sensor (200Hz) → Buffer (200 samples) → Networks (1Hz) → EKF (200Hz) → Output
```

### File Formats
- **IMU**: `timestamp, gx, gy, gz, ax, ay, az`
- **Ground Truth**: `timestamp, px, py, pz, qw, qx, qy, qz`
- **Throttle**: `timestamp, throttle` (optional)

---

## 🆘 Getting Help

### Common Issues

**Training not working?**
→ Check [FULL_PIPELINE_GUIDE.md](FULL_PIPELINE_GUIDE.md) Troubleshooting

**Sensor not connecting?**
→ Check [QUICKSTART.md](QUICKSTART.md) Sensor Issues

**Poor accuracy?**
→ Check ground truth quality, increase training epochs

**High latency?**
→ Reduce window_size, use GPU

---

## 📊 Performance Targets

| Metric | Target |
|--------|--------|
| Training time | 2-4 hours |
| Inference rate | 200 Hz |
| Position error | 0.1-0.5 m (60s) |
| Velocity error | 0.05-0.2 m/s |
| Latency | 5-10 ms |

---

## 🎯 Next Steps

1. **Read** [QUICKSTART.md](QUICKSTART.md) (5 min)
2. **Test** demo: `python realtime_pipeline_demo.py`
3. **Prepare** your dataset
4. **Train** your model
5. **Deploy** with real sensor

---

## 📞 Support

For issues:
1. Check relevant documentation section
2. Review troubleshooting guides
3. Verify file formats
4. Test components separately

---

## 🎉 You're Ready!

Everything you need is documented and ready to use. Start with [QUICKSTART.md](QUICKSTART.md) and you'll be running real-time IMU odometry within a day!

**Happy coding! 🚀**
