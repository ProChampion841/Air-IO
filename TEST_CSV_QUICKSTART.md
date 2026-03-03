# Test Your IMU CSV - Quick Reference

## One Command to Test

```bash
python test_imu_csv.py --csv YOUR_IMU_DATA.csv --model YOUR_MODEL.pth
```

---

## CSV Format

```csv
# timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
1234567890.000, 0.01, -0.02, 1.25, 0.05, -0.03, 9.81
1234567890.005, 0.01, -0.02, 1.26, 0.05, -0.03, 9.82
```

**Units**: gyro (rad/s), acc (m/s²)

---

## What You Get

1. **Console output**: Statistics and final state
2. **Plots**: 6 visualizations (3D trajectory, velocity, etc.)
3. **Saved files**: 
   - `test_results/trajectory_plot.png`
   - `test_results/odometry_results.npy`

---

## Example

```bash
# Test with example data
python test_imu_csv.py \
    --csv example_imu_data.csv \
    --model experiments/blackbird/motion_body_rot/best_model.pth
```

---

## Full Documentation

See **TEST_IMU_CSV_GUIDE.md** for:
- Detailed usage
- Troubleshooting
- Advanced options
- Performance tuning
