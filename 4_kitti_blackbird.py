#!/usr/bin/env python3
"""
Convert a KITTI Odometry sequence (poses + times) into a Blackbird-style sequence folder.

Blackbird (as in the attached sample) expects:
  - groundTruthPoses.csv : NO header
      timestamp_us, p_x, p_y, p_z, q_x, q_y, q_z, q_w
    where timestamp_us is integer microseconds (epoch-like).
  - imu_data.csv : header row
      # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    (timestamps are float seconds)
  - thrust_data.csv : header row
      # timestamp, thrust_1, thrust_2, thrust_3, thrust_4

Because KITTI does not include IMU/thrust for the odometry benchmark, this script can
either create empty files with only headers (default) or create zero-filled rows aligned
to the pose timestamps.

Usage:
  python kitti_to_blackbird.py \
      --kitti_zip /path/to/Kitti.zip \
      --seq 00 \
      --out_dir /path/to/output \
      --bb_seq_name maxSpeed5p0 \
      --epoch_start 1525745925.0 \
      --fill_dummy_sensors

Notes:
- The pose conversion uses KITTI poses (3x4) directly:
    position = translation (tx, ty, tz)
    quaternion = rotation matrix -> (qx, qy, qz, qw)
- Coordinate-frame differences between KITTI and Blackbird are NOT automatically handled.
  If you need a frame conversion (e.g., KITTI camera to ENU/NED), add a fixed rotation.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    Numerically stable, returns unit quaternion.
    """
    # Based on the classic trace method, with branch handling.
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # Normalize (important if R isn't perfectly orthonormal due to numeric noise)
    q /= np.linalg.norm(q) + 1e-12
    return q


def read_kitti_poses_from_zip(z: zipfile.ZipFile, seq: str) -> np.ndarray:
    """
    Reads poses from either:
      - top-level "{seq}.txt" (common in many packaged datasets), OR
      - "{seq}/poses.txt" (some variants).
    Returns: (N, 12) float array.
    """
    candidates = [f"{seq}.txt", f"{seq}/poses.txt"]
    txt_name = None
    for c in candidates:
        if c in z.namelist():
            txt_name = c
            break
    if txt_name is None:
        raise FileNotFoundError(f"Could not find poses file in zip. Tried: {candidates}")

    raw = z.open(txt_name).read().decode("utf-8", errors="ignore").strip().splitlines()
    if not raw:
        raise ValueError(f"Poses file {txt_name} is empty.")

    data = []
    for i, line in enumerate(raw):
        parts = line.strip().split()
        if len(parts) != 12:
            raise ValueError(f"Line {i} in {txt_name} expected 12 floats, got {len(parts)}.")
        data.append([float(x) for x in parts])

    return np.asarray(data, dtype=np.float64)


def read_kitti_times_from_zip(z: zipfile.ZipFile, seq: str) -> np.ndarray:
    """
    Reads times from "{seq}/times.txt" (KITTI odometry format).
    Returns: (N,) float seconds-from-start.
    """
    name = f"{seq}/times.txt"
    if name not in z.namelist():
        raise FileNotFoundError(f"Could not find {name} in zip.")

    raw = z.open(name).read().decode("utf-8", errors="ignore").strip().splitlines()
    if not raw:
        raise ValueError(f"Times file {name} is empty.")
    return np.asarray([float(x) for x in raw], dtype=np.float64)


def write_blackbird_sequence(
    poses_12: np.ndarray,
    times_s: np.ndarray,
    out_seq_dir: str,
    epoch_start_s: float,
    fill_dummy_sensors: bool,
) -> None:
    """
    Create:
      - groundTruthPoses.csv (no header)
      - imu_data.csv
      - thrust_data.csv
    """
    os.makedirs(out_seq_dir, exist_ok=True)

    if poses_12.shape[0] != times_s.shape[0]:
        raise ValueError(f"poses rows ({poses_12.shape[0]}) != times rows ({times_s.shape[0]})")

    # Convert to Blackbird GT rows
    rows = []
    for i in range(poses_12.shape[0]):
        P = poses_12[i].reshape(3, 4)
        R = P[:, :3]
        t = P[:, 3]
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R)

        ts_s = float(epoch_start_s + times_s[i])
        ts_us = int(round(ts_s * 1_000_000.0))  # microseconds, integer
        rows.append([ts_us, t[0], t[1], t[2], qx, qy, qz, qw])

    gt_path = os.path.join(out_seq_dir, "groundTruthPoses.csv")
    gt_df = pd.DataFrame(rows)
    # Match the sample: comma-separated, no header, no index.
    # Use 6 decimals to resemble typical Blackbird formatting; timestamps stay integer.
    gt_df.to_csv(
        gt_path,
        header=False,
        index=False,
        float_format="%.6f",
    )

    # IMU + thrust: either headers only, or zero-filled rows aligned to pose timestamps.
    imu_cols = ["# timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    thrust_cols = ["# timestamp", "thrust_1", "thrust_2", "thrust_3", "thrust_4"]

    if fill_dummy_sensors:
        ts = (epoch_start_s + times_s).astype(np.float64)
        imu_df = pd.DataFrame(
            {
                "# timestamp": ts,
                "acc_x": np.zeros_like(ts),
                "acc_y": np.zeros_like(ts),
                "acc_z": np.zeros_like(ts),
                "gyro_x": np.zeros_like(ts),
                "gyro_y": np.zeros_like(ts),
                "gyro_z": np.zeros_like(ts),
            }
        )
        thrust_df = pd.DataFrame(
            {
                "# timestamp": ts,
                "thrust_1": np.zeros_like(ts),
                "thrust_2": np.zeros_like(ts),
                "thrust_3": np.zeros_like(ts),
                "thrust_4": np.zeros_like(ts),
            }
        )
    else:
        imu_df = pd.DataFrame(columns=imu_cols)
        thrust_df = pd.DataFrame(columns=thrust_cols)

    imu_path = os.path.join(out_seq_dir, "imu_data.csv")
    thrust_path = os.path.join(out_seq_dir, "thrust_data.csv")

    imu_df.to_csv(imu_path, index=False)
    thrust_df.to_csv(thrust_path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kitti_zip", required=True, help="Path to a KITTI zip (containing seq folder + times.txt).")
    p.add_argument("--seq", default="00", help="KITTI sequence id, e.g. 00.")
    p.add_argument("--out_dir", required=True, help="Output directory to write the Blackbird-style sequence.")
    p.add_argument("--bb_seq_name", default="maxSpeed5p0", help="Name of the Blackbird sequence folder to create.")
    p.add_argument(
        "--epoch_start",
        type=float,
        default=1525745925.0,
        help="Synthetic epoch start time in seconds. groundTruth timestamps become int microseconds.",
    )
    p.add_argument(
        "--fill_dummy_sensors",
        action="store_true",
        help="If set, generate imu/thrust rows (all zeros) aligned to pose timestamps. Otherwise headers only.",
    )
    args = p.parse_args()

    with zipfile.ZipFile(args.kitti_zip, "r") as z:
        poses_12 = read_kitti_poses_from_zip(z, args.seq)
        times_s = read_kitti_times_from_zip(z, args.seq)

    out_seq_dir = os.path.join(args.out_dir, args.bb_seq_name)
    write_blackbird_sequence(
        poses_12=poses_12,
        times_s=times_s,
        out_seq_dir=out_seq_dir,
        epoch_start_s=args.epoch_start,
        fill_dummy_sensors=args.fill_dummy_sensors,
    )

    print(f"Wrote Blackbird-style sequence to: {out_seq_dir}")


if __name__ == "__main__":
    main()
