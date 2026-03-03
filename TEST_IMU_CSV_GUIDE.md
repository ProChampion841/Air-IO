# Testing IMU CSV Files with Your Model

## Quick Start

### 1. Prepare Your CSV File

Your CSV file should have one of these formats:

**Format 1 (with timestamp):**
```csv
# timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
1234567890.123, 0.01, -0.02, 1.25, 0.05, -0.03, 9.81
1234567890.128, 0.01, -0.02, 1.26, 0.05, -0.03, 9.82
```

**Format 2 (without timestamp, assumes 200Hz):**
```csv
# gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
0.01, -0.02, 1.25, 0.05, -0.03, 9.81
0.01, -0.02, 1.26, 0.05, -0.03, 9.82
```

**Units:**
- Gyroscope: rad/s
- Accelerometer: m/s²

### 2. Run Test

```bash
python test_imu_csv.py \
    --csv path/to/your/imu_data.csv \
    --model experiments/mydataset/motion_body_rot/best_model.pth
```

### 3. View Results

The script will:
- ✅ Load your CSV file
- ✅ Run your trained model
- ✅ Fuse with EKF
- ✅ Display trajectory plots
- ✅ Save results to `test_results/`

---

## Command Options

```bash
python test_imu_csv.py \
    --csv IMU_DATA.csv \              # Required: Your IMU CSV file
    --model MODEL.pth \                # Required: Your trained model
    --window-size 200 \                # Optional: Window size (default: 200)
    --no-save                          # Optional: Don't save results
```

---

## Examples

### Example 1: Basic Test
```bash
python test_imu_csv.py \
    --csv data/test_flight.csv \
    --model experiments/blackbird/motion_body_rot/best_model.pth
```

### Example 2: Larger Window
```bash
python test_imu_csv.py \
    --csv data/test_flight.csv \
    --model experiments/blackbird/motion_body_rot/best_model.pth \
    --window-size 1000
```

### Example 3: Quick Test (no save)
```bash
python test_imu_csv.py \
    --csv data/test_flight.csv \
    --model experiments/blackbird/motion_body_rot/best_model.pth \
    --no-save
```

---

## Output

### Console Output
```
======================================================================
Testing IMU CSV with Trained Model
======================================================================

[1/5] Loading IMU data...
✓ Loaded 10000 samples
  Duration: 50.00s
  Average rate: 200.0 Hz

[2/5] Loading trained model...
✓ Loaded codewithrot model from experiments/.../best_model.pth

[3/5] Initializing EKF...

[4/5] Processing IMU data...
100%|████████████████████████████| 10000/10000 [00:45<00:00]

[5/5] Results:
----------------------------------------------------------------------
Total samples: 10000
Duration: 50.00s

Final state:
  Position: [12.345, 5.678, 1.234] m
  Velocity: [0.123, 0.456, 0.012] m/s
  Total distance: 45.67 m

Generating plots...
✓ Plot saved to test_results/trajectory_plot.png
✓ Results saved to test_results/odometry_results.npy

======================================================================
Testing completed!
======================================================================
```

### Generated Files

```
test_results/
├── trajectory_plot.png       # 6 subplots visualization
└── odometry_results.npy      # Numerical results
```

### Plots Include:
1. 3D trajectory
2. 2D trajectory (top view)
3. Position vs time (X, Y, Z)
4. Velocity vs time (Vx, Vy, Vz)
5. Speed vs time
6. Distance traveled vs time

---

## CSV File Requirements

### Minimum Requirements
- At least 200 samples (for window_size=200)
- Consistent sampling rate
- Valid numerical data

### Recommended
- 1000+ samples for meaningful trajectory
- 200 Hz sampling rate
- Clean data (no NaN or inf values)

### Data Quality
- Gyroscope bias: < 0.1 rad/s
- Accelerometer bias: < 0.5 m/s²
- Noise level: Reasonable for IMU sensor

---

## Troubleshooting

### Error: "Unexpected CSV format"
- Check your CSV has 6 or 7 columns
- Ensure no extra commas or spaces
- Remove any non-numeric rows (except header starting with #)

### Error: "Model checkpoint not found"
- Verify model path is correct
- Check file extension is .pth
- Ensure model was trained successfully

### Error: "CUDA out of memory"
- Reduce window_size: `--window-size 100`
- Or force CPU: Set `device = torch.device('cpu')` in script

### Poor Results (Large drift)
- Check IMU data quality
- Verify units (rad/s for gyro, m/s² for acc)
- Ensure model was trained on similar data
- Try tuning EKF Q and R matrices

---

## Advanced Usage

### Modify EKF Parameters

Edit `test_imu_csv.py`:

```python
# Line ~120
Q = torch.eye(12, dtype=torch.float64) * 1e-4  # Process noise
R = torch.eye(3, dtype=torch.float64) * 1e-2   # Measurement noise
```

Increase Q for more trust in measurements, decrease for more trust in model.

### Change Window Size

```bash
# Smaller window = faster but less accurate
python test_imu_csv.py --csv data.csv --model model.pth --window-size 100

# Larger window = slower but more accurate
python test_imu_csv.py --csv data.csv --model model.pth --window-size 1000
```

### Load and Analyze Saved Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
results = np.load('test_results/odometry_results.npy', allow_pickle=True).item()

timestamps = results['timestamps']
positions = results['positions']
velocities = results['velocities']

# Your custom analysis
print(f"Max speed: {np.max(np.linalg.norm(velocities, axis=1)):.2f} m/s")
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|---------------|
| Processing speed | 200-500 samples/sec |
| Position drift | 0.1-0.5 m per 60s |
| Velocity error | 0.05-0.2 m/s |
| Memory usage | ~100 MB |

---

## Next Steps

1. **Test with your CSV**: Run the script with your data
2. **Analyze results**: Check trajectory makes sense
3. **Tune parameters**: Adjust EKF Q/R if needed
4. **Compare with ground truth**: If available
5. **Deploy real-time**: Use `realtime_deployment.py` for live sensor

---

## Related Files

- `realtime_deployment.py` - For real-time sensor testing
- `realtime_sensor_interface.py` - For live IMU connection
- `MODEL_IO_SUMMARY.md` - Model input/output specs
