import math
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def euler_xyz_to_quat_wxyz(roll, pitch, yaw):
    """
    Convert roll/pitch/yaw (rad) to quaternion (w,x,y,z).
    Assumes roll about X, pitch about Y, yaw about Z, applied in ZYX order.
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qw, qx, qy, qz


def latlon_to_enu_m(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m):
    """
    Simple local tangent plane (ENU) conversion using equirectangular approximation.
    Good for small areas.
    """
    R = 6378137.0  # meters
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    north = dlat * R
    east = dlon * R * math.cos(lat0)
    up = alt_m - alt0_m
    return east, north, up


# ----------------------------
# Main conversion
# ----------------------------
def convert_to_blackbird(
    in_csv="filtered_mock_imu_gps.csv",
    out_imu="imu_data.csv",
    out_thrust="thrust_data.csv",
    out_gt="groundTruthPoses.csv",
    thrust_scale=0.011,      # thrust_3 = -Throttle * thrust_scale
    accel_scale=1.0,         # set to 9.81 if your accel is in "g"
    gyro_scale=1.0           # set if your gyro needs scaling
):
    df = pd.read_csv(in_csv)

    # Ensure sorted by time
    df = df.sort_values("NanoTimeStamp").reset_index(drop=True)

    # Timestamps
    ts_ns = df["NanoTimeStamp"].astype("int64")
    ts_s = ts_ns / 1e9
    ts_us = (ts_ns / 1000).astype("int64")  # groundTruthPoses uses microseconds-like in your sample

    # ----------------------------
    # imu_data.csv (with header)
    # ----------------------------
    imu = pd.DataFrame({
        "# timestamp": ts_s.astype("float64"),
        " acc_x": (df["AcclX"] * accel_scale).astype("float64"),
        " acc_y": (df["AcclY"] * accel_scale).astype("float64"),
        " acc_z": (df["AcclZ"] * accel_scale).astype("float64"),
        " gyro_x": (df["GyroX"] * gyro_scale).astype("float64"),
        " gyro_y": (df["GyroY"] * gyro_scale).astype("float64"),
        " gyro_z": (df["GyroZ"] * gyro_scale).astype("float64"),
    })
    imu.to_csv(out_imu, index=False)

    # ----------------------------
    # thrust_data.csv (with header)
    # Match your example: thrust_1, thrust_2 = 0, thrust_3 negative, thrust_4 = 0
    # ----------------------------
    thrust = pd.DataFrame({
        "# timestamp": ts_s.astype("float64"),
        " thrust_1": 0.0,
        " thrust_2": 0.0,
        " thrust_3": -(df["Throttle"].astype("float64") * thrust_scale),
        "thrust_4": 0.0,  # NOTE: no leading space, matches your attached file
    })
    thrust.to_csv(out_thrust, index=False)

    # ----------------------------
    # groundTruthPoses.csv (NO header)
    # Use GPS -> local ENU position, and GPSNavEUL* -> quaternion
    # ----------------------------
    lat0 = float(df.loc[0, "GPSLat"])
    lon0 = float(df.loc[0, "GPSLon"])
    alt0 = float(df.loc[0, "GPSAlt"])

    xs, ys, zs, qws, qxs, qys, qzs = [], [], [], [], [], [], []
    for i in range(len(df)):
        lat = float(df.loc[i, "GPSLat"])
        lon = float(df.loc[i, "GPSLon"])
        alt = float(df.loc[i, "GPSAlt"])
        x, y, z = latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0)
        xs.append(x)
        ys.append(y)
        zs.append(z)

        # Assumption: GPSNavEULX/Y/Z are roll/pitch/yaw in radians
        roll = float(df.loc[i, "GPSNavEULX"])
        pitch = float(df.loc[i, "GPSNavEULY"])
        yaw = float(df.loc[i, "GPSNavEULZ"])
        qw, qx, qy, qz = euler_xyz_to_quat_wxyz(roll, pitch, yaw)
        qws.append(qw); qxs.append(qx); qys.append(qy); qzs.append(qz)

    gt = pd.DataFrame({
        "timestamp_us": ts_us,
        "x": xs,
        "y": ys,
        "z": zs,
        "qw": qws,
        "qx": qxs,
        "qy": qys,
        "qz": qzs,
    })

    # Write with NO header (to match your attached Blackbird groundTruthPoses.csv)
    gt.to_csv(out_gt, index=False, header=False)

    print("Wrote:")
    print(" -", out_imu)
    print(" -", out_thrust)
    print(" -", out_gt)


if __name__ == "__main__":
    convert_to_blackbird()