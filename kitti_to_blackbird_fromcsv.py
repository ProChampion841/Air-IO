"""
Convert KITTI one-sequence data to Blackbird format (Air-IO compatible),
matching the EXACT CSV schemas observed in the provided Blackbird sample.

Blackbird sample schemas (from your attached Blackbird sequence):
- imu_data.csv:
    # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    <float_seconds>,<acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>
- groundTruthPoses.csv:
    <int_microseconds>,<x>,<y>,<z>,<qw>,<qx>,<qy>,<qz>
- thrust_data.csv:
    # timestamp, thrust_1, thrust_2, thrust_3,thrust_4   (comment line kept)
    <float_seconds>,<thrust_1>,<thrust_2>,<thrust_3>     (sample has 3 thrust cols)

Notes:
- KITTI times.txt is seconds-from-start (float). We keep it relative.
- Poses get timestamps as int microseconds (relative): int(t * 1e6).
- IMU/thrust timestamps are float seconds with 6 decimals: f"{t:.6f}"
"""

import os
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation
import argparse

G0 = 9.80665  # m/s^2 per g

def read_kitti_poses(pose_file: str) -> np.ndarray:
    """Read KITTI poses (each line is 12 floats => 3x4). Returns (N,4,4)."""
    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                raise ValueError(f"Pose line does not have 12 floats: {pose_file}")
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return np.stack(poses, axis=0)

def read_kitti_times(times_file: str) -> np.ndarray:
    """Read KITTI timestamps (seconds, float)."""
    times = np.loadtxt(times_file, dtype=np.float64)
    if times.ndim == 0:
        times = np.array([float(times)], dtype=np.float64)
    return times

def pose_to_position_quaternion(T: np.ndarray):
    """Convert 4x4 to position (x,y,z) and quaternion (qw,qx,qy,qz)."""
    t = T[:3, 3]
    Rm = T[:3, :3]
    q_xyzw = Rotation.from_matrix(Rm).as_quat()  # [x,y,z,w]
    qx, qy, qz, qw = q_xyzw
    return t, (qw, qx, qy, qz)

def similarity_transform_pose(T: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Apply coordinate-frame change with rotation C (3x3) via similarity:
      T_new = [C 0;0 1] * T * [C^T 0;0 1]
    """
    Tn = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Tn[:3, :3] = C @ R @ C.T
    Tn[:3, 3]  = C @ t
    return Tn

def find_sequence_files(kitti_root: str, sequence: str):
    """
    Supports your uploaded layout:
      <kitti_root>/<sequence>.mat
      <kitti_root>/<sequence>.txt
      <kitti_root>/<sequence>/times.txt
    And also a more "classic" layout if present.
    """
    # Your zip layout
    imu_file  = os.path.join(kitti_root, f"{sequence}.mat")
    pose_file = os.path.join(kitti_root, f"{sequence}.txt")
    times_file = os.path.join(kitti_root, sequence, "times.txt")

    if os.path.isfile(imu_file) and os.path.isfile(pose_file) and os.path.isfile(times_file):
        return imu_file, pose_file, times_file

    # Fallback to original script layout
    imu_file2  = os.path.join(kitti_root, "imus", f"{sequence}.mat")
    pose_file2 = os.path.join(kitti_root, "poses", f"{sequence}.txt")
    times_file2 = os.path.join(kitti_root, "sequences", sequence, "times.txt")

    if os.path.isfile(imu_file2) and os.path.isfile(pose_file2) and os.path.isfile(times_file2):
        return imu_file2, pose_file2, times_file2

    raise FileNotFoundError(
        f"Could not locate files for sequence {sequence} under {kitti_root}.\n"
        f"Tried:\n"
        f"  {imu_file}\n  {pose_file}\n  {times_file}\n"
        f"and:\n"
        f"  {imu_file2}\n  {pose_file2}\n  {times_file2}\n"
    )

def convert_kitti_sequence(kitti_root: str, sequence: str, output_root: str,
                           imu_samples_per_pose: int = 10,
                           accel_to_g: bool = True,
                           assume_kitti_imu_is_flu_zdown: bool = True,
                           assume_kitti_pose_is_cam_frame: bool = True):
    """
    - accel_to_g: convert accel from m/s^2 to g units
    - assume_kitti_imu_is_flu_zdown: interpret imu_data_interp accel as (x forward, y left, z down)
      and convert to FLU (z up) by flipping z (and wz).
    - assume_kitti_pose_is_cam_frame: interpret pose txt as camera frame (x right, y down, z forward)
      and convert to FLU with similarity transform.
    """
    print(f"Converting sequence {sequence}...")

    imu_file, pose_file, times_file = find_sequence_files(kitti_root, sequence)

    # Load IMU data
    mat = sio.loadmat(imu_file)
    if "imu_data_interp" not in mat:
        raise KeyError(f"'imu_data_interp' not found in {imu_file}")
    imu_raw = np.asarray(mat["imu_data_interp"], dtype=np.float64)  # (N,6) [ax,ay,az, wx,wy,wz]

    # --- IMU frame / units to match Blackbird-like expectations ---
    ax, ay, az, wx, wy, wz = imu_raw.T

    # Convert accel units to g (matches Blackbird magnitudes much better than m/s^2)
    if accel_to_g:
        ax, ay, az = ax / G0, ay / G0, az / G0

    # Convert to FLU (x forward, y left, z up)
    # If KITTI IMU is already (x forward, y left, z down): just flip z and wz.
    if assume_kitti_imu_is_flu_zdown:
        az = -az
        wz = -wz
    else:
        # If instead it is in KITTI camera frame (x right, y down, z forward),
        # map to FLU: [x,y,z] = [z, -x, -y]
        ax, ay, az = az, -ax, -ay
        wx, wy, wz = wz, -wx, -wy

    imu_data = np.stack([ax, ay, az, wx, wy, wz], axis=1)  # (N,6) [acc..., gyro...]

    # Load poses and timestamps
    poses_raw = read_kitti_poses(pose_file)
    times = read_kitti_times(times_file)

    if len(poses_raw) != len(times):
        raise ValueError(f"poses ({len(poses_raw)}) != times ({len(times)}) for seq {sequence}")

    # Pose transform: camera frame -> FLU, via similarity transform
    if assume_kitti_pose_is_cam_frame:
        # KITTI cam: x right, y down, z forward
        # FLU: x forward, y left, z up
        C = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0]
        ], dtype=np.float64)
        poses = np.stack([similarity_transform_pose(T, C) for T in poses_raw], axis=0)
    else:
        poses = poses_raw

    # --- Build IMU timestamps (float seconds) ---
    # times are pose times. IMU is ~10 samples between pose times in your KITTI sample.
    imu_times = []
    for i in range(len(times) - 1):
        t0, t1 = float(times[i]), float(times[i + 1])
        imu_times.extend(np.linspace(t0, t1, imu_samples_per_pose, endpoint=False).tolist())
    imu_times.append(float(times[-1]))  # last
    imu_times = np.array(imu_times, dtype=np.float64)

    # Ensure length matches IMU samples (trim/pad safely)
    N = imu_data.shape[0]
    if len(imu_times) < N:
        # pad by repeating last dt if needed
        if len(times) >= 2:
            dt = float(times[-1] - times[-2]) / imu_samples_per_pose
        else:
            dt = 0.01
        extra = N - len(imu_times)
        imu_times = np.concatenate([imu_times, imu_times[-1] + dt * np.arange(1, extra + 1)], axis=0)
    imu_times = imu_times[:N]

    # Output directory
    out_dir = os.path.join(output_root, f"seq_{sequence}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Write imu_data.csv (Blackbird schema) ---
    imu_path = os.path.join(out_dir, "imu_data.csv")
    with open(imu_path, "w") as f:
        f.write("# timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z\n")
        for t, (ax, ay, az, wx, wy, wz) in zip(imu_times, imu_data):
            # timestamp float seconds with 6 decimals like the sample
            # values use 15 sig figs (stable + precise)
            f.write(
                f"{t:.6f},"
                f"{ax:.15g},{ay:.15g},{az:.15g},"
                f"{wx:.15g},{wy:.15g},{wz:.15g}\n"
            )

    # --- Write groundTruthPoses.csv (Blackbird schema) ---
    pose_path = os.path.join(out_dir, "groundTruthPoses.csv")
    with open(pose_path, "w") as f:
        for t, T in zip(times, poses):
            ts_us = int(round(float(t) * 1e6))  # int microseconds
            pos, (qw, qx, qy, qz) = pose_to_position_quaternion(T)
            x, y, z = pos
            # Blackbird sample uses ~6 decimals
            f.write(f"{ts_us},{x:.6f},{y:.6f},{z:.6f},{qw:.6f},{qx:.6f},{qy:.6f},{qz:.6f}\n")

    # --- Write thrust_data.csv (match Blackbird sample: float seconds + 3 thrust cols) ---
    thrust_path = os.path.join(out_dir, "thrust_data.csv")
    with open(thrust_path, "w") as f:
        # Keep the same comment header style as your Blackbird sample
        f.write("# timestamp, thrust_1, thrust_2, thrust_3,thrust_4\n")
        for t in times:
            f.write(f"{float(t):.6f},0.0,0.0,0.0\n")

    print(f"  Wrote: {imu_path}")
    print(f"  Wrote: {pose_path}")
    print(f"  Wrote: {thrust_path}")
    print(f"  Converted {len(imu_data)} IMU samples and {len(poses)} poses\n")

def main():
    parser = argparse.ArgumentParser(description="Convert KITTI to Blackbird format (matching sample schema)")
    parser.add_argument("--kitti_root", type=str, required=True, help="Path to KITTI root")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory")
    parser.add_argument("--sequences", type=str, nargs="+", default=["00"], help="Sequences to convert")

    # knobs (leave defaults unless you know you need to change)
    parser.add_argument("--imu_samples_per_pose", type=int, default=10, help="IMU samples per pose interval")
    parser.add_argument("--no_accel_to_g", action="store_true", help="Do not convert accel m/s^2 to g")
    parser.add_argument("--imu_is_cam_frame", action="store_true",
                        help="Treat imu_data_interp as KITTI camera frame (x right,y down,z forward)")
    parser.add_argument("--pose_is_not_cam_frame", action="store_true",
                        help="Do not transform poses from KITTI camera frame to FLU")

    args = parser.parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    for seq in args.sequences:
        convert_kitti_sequence(
            kitti_root=args.kitti_root,
            sequence=seq,
            output_root=args.output_root,
            imu_samples_per_pose=args.imu_samples_per_pose,
            accel_to_g=not args.no_accel_to_g,
            assume_kitti_imu_is_flu_zdown=not args.imu_is_cam_frame,
            assume_kitti_pose_is_cam_frame=not args.pose_is_not_cam_frame
        )

if __name__ == "__main__":
    main()