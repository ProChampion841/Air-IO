import csv
import random

def generate_mock_csv(
    out_path="mock_imu_gps.csv",
    steps=100000,
    start_timestamp=65135462456640,
    step_ns=100_000_000,          # 0.1 seconds in nanoseconds
    gps_update_every=10,
    gps_nav_eul_zero_steps=1000,  # keep EUL at 0 for first 1000 steps
    seed=42
):
    random.seed(seed)

    headers = [
        "NanoTimeStamp",
        "GyroX", "GyroY", "GyroZ",
        "AcclX", "AcclY", "AcclZ",
        "GPSLat", "GPSLon", "GPSAlt",
        "GPSNavEULX", "GPSNavEULY", "GPSNavEULZ",
        "Throttle"
    ]

    # Starting GPS (use your sample-ish values)
    gps_lat = 38.79497
    gps_lon = 125.83960
    gps_alt = 55.93361

    # Throttle behavior (example: mostly high with a little noise)
    throttle = 1000

    # Simple drifting signals for IMU
    gyro_bias = [0.0, 0.0, 0.0]
    accl_bias = [0.0, 0.0, -0.8]  # bias Z a bit negative like your sample

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(steps):
            ts = start_timestamp + i * step_ns

            # Update GPS every N steps (random walk)
            if i % gps_update_every == 0 and i != 0:
                gps_lat += random.uniform(-0.00003, 0.00003)
                gps_lon += random.uniform(-0.00003, 0.00003)
                gps_alt += random.uniform(-0.02, 0.02)

            # Slightly drift biases over time
            for k in range(3):
                gyro_bias[k] += random.uniform(-0.0005, 0.0005)
                accl_bias[k] += random.uniform(-0.0008, 0.0008)

            # Gyro readings
            gyro_x = gyro_bias[0] + random.gauss(0, 0.25)
            gyro_y = gyro_bias[1] + random.gauss(0, 0.25)
            gyro_z = gyro_bias[2] + random.gauss(0, 0.25)

            # Accelerometer readings
            accl_x = accl_bias[0] + random.gauss(0, 0.20)
            accl_y = accl_bias[1] + random.gauss(0, 0.20)
            accl_z = accl_bias[2] + random.gauss(0, 0.25)

            # GPSNavEUL: 0 for first gps_nav_eul_zero_steps
            if i < gps_nav_eul_zero_steps:
                eul_x = 0.0
                eul_y = 0.0
                eul_z = 0.0
            else:
                # If you ever extend beyond 1000 steps, this gives small nonzero motion
                eul_x = random.uniform(-0.01, 0.01)
                eul_y = random.uniform(-0.5, 0.5)
                eul_z = random.uniform(-0.01, 0.01)

            # Throttle example: keep around 1000, occasionally bump like your sample third row
            if i % 200 == 0 and i != 0:
                throttle = min(2000, throttle + 300)
            else:
                throttle = max(0, min(2000, throttle + random.randint(-5, 5)))

            writer.writerow([
                ts,
                f"{gyro_x:.5f}", f"{gyro_y:.5f}", f"{gyro_z:.5f}",
                f"{accl_x:.6f}", f"{accl_y:.5f}", f"{accl_z:.5f}",
                f"{gps_lat:.5f}", f"{gps_lon:.5f}", f"{gps_alt:.5f}",
                f"{eul_x:.6f}", f"{eul_y:.6f}", f"{eul_z:.6f}",
                throttle
            ])

    print(f"Wrote {steps} rows to {out_path}")

if __name__ == "__main__":
    generate_mock_csv()