"""
Convert KITTI dataset to Blackbird format for Air-IO
KITTI format:
- imus/*.mat: IMU data (N, 6) [ax, ay, az, wx, wy, wz] at 10Hz
- poses/*.txt: Poses as 3x4 transformation matrices (one per line)
- sequences/*/times.txt: Timestamps

Blackbird format:
- imu_data.csv: timestamp,wx,wy,wz,ax,ay,az
- groundTruthPoses.csv: timestamp,x,y,z,qw,qx,qy,qz
- thrust_data.csv: timestamp,thrust (dummy for KITTI)
"""

import os
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation
import argparse

def read_kitti_poses(pose_file):
    """Read KITTI poses (3x4 transformation matrices)"""
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            # Reshape to 3x4 matrix
            T = np.array(values).reshape(3, 4)
            # Convert to 4x4
            T_full = np.eye(4)
            T_full[:3, :] = T
            poses.append(T_full)
    return np.array(poses)

def read_kitti_times(times_file):
    """Read KITTI timestamps"""
    times = []
    with open(times_file, 'r') as f:
        for line in f:
            times.append(float(line.strip()))
    return np.array(times)

def pose_to_position_quaternion(T):
    """Convert 4x4 transformation matrix to position and quaternion"""
    position = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [x, y, z, w]
    return position, quaternion

def convert_kitti_sequence(kitti_root, sequence, output_root):
    """Convert one KITTI sequence to Blackbird format"""
    print(f"Converting sequence {sequence}...")
    
    # Read KITTI data
    imu_file = os.path.join(kitti_root, 'imus', f'{sequence}.mat')
    pose_file = os.path.join(kitti_root, 'poses', f'{sequence}.txt')
    times_file = os.path.join(kitti_root, 'sequences', sequence, 'times.txt')
    
    # Load IMU data - KITTI format: [ax, ay, az, wx, wy, wz] in camera frame
    imu_data_raw = sio.loadmat(imu_file)['imu_data_interp']  # (N, 6)
    
    # KITTI camera frame: x=right, y=down, z=forward
    # Convert to standard IMU frame: x=forward, y=left, z=up
    # Transformation: [x_imu, y_imu, z_imu] = [z_cam, -x_cam, -y_cam]
    imu_data = np.zeros_like(imu_data_raw)
    """
    # Accelerometer: [ax, ay, az]
    imu_data[:, 0] = imu_data_raw[:, 2]   # ax_imu = az_cam (forward)
    imu_data[:, 1] = -imu_data_raw[:, 0]  # ay_imu = -ax_cam (left)
    imu_data[:, 2] = -imu_data_raw[:, 1]  # az_imu = -ay_cam (up)
    # Gyroscope: [wx, wy, wz]
    imu_data[:, 3] = imu_data_raw[:, 5]   # wx_imu = wz_cam
    imu_data[:, 4] = -imu_data_raw[:, 3]  # wy_imu = -wx_cam
    imu_data[:, 5] = -imu_data_raw[:, 4]  # wz_imu = -wy_cam
    """
    # Accelerometer: [ax, ay, az]
    imu_data[:, 0] = imu_data_raw[:, 0]
    imu_data[:, 1] = imu_data_raw[:, 1]
    imu_data[:, 2] = imu_data_raw[:, 2]
    # Gyroscope: [wx, wy, wz]
    imu_data[:, 3] = imu_data_raw[:, 3]
    imu_data[:, 4] = imu_data_raw[:, 4]
    imu_data[:, 5] = imu_data_raw[:, 5]
    
    # Load poses - also need coordinate transformation
    poses_raw = read_kitti_poses(pose_file)
    
    # Transform poses from camera frame to IMU frame
    T_cam_to_imu = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    poses = np.array([T_cam_to_imu @ pose for pose in poses_raw])
    
    # Load timestamps
    times = read_kitti_times(times_file)
    
    # KITTI IMU is at 10Hz, so 10 IMU samples per pose
    IMU_FREQ = 10
    
    # Create output directory
    output_dir = os.path.join(output_root, f'seq_{sequence}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate IMU timestamps (interpolate between pose timestamps)
    imu_times = []
    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i + 1]
        imu_times.extend(np.linspace(t_start, t_end, IMU_FREQ, endpoint=False))
    # Add last timestamp
    if len(times) > 0:
        imu_times.append(times[-1])
    imu_times = np.array(imu_times[:len(imu_data)])
    
    # Write IMU data (Blackbird format: timestamp,wx,wy,wz,ax,ay,az)
    imu_output = os.path.join(output_dir, 'imu_data.csv')
    with open(imu_output, 'w') as f:
        for i in range(len(imu_data)):
            timestamp = int(imu_times[i] * 1e6)  # Convert to microseconds
            ax, ay, az, wx, wy, wz = imu_data[i]
            f.write(f"{timestamp},{wx:.10f},{wy:.10f},{wz:.10f},{ax:.10f},{ay:.10f},{az:.10f}\n")
    
    # Write ground truth poses (Blackbird format: timestamp,x,y,z,qw,qx,qy,qz)
    pose_output = os.path.join(output_dir, 'groundTruthPoses.csv')
    with open(pose_output, 'w') as f:
        for i in range(len(poses)):
            timestamp = int(times[i] * 1e6)
            position, quaternion = pose_to_position_quaternion(poses[i])
            x, y, z = position

            # Keep quaternion in [qx, qy, qz, qw] format for scipy
            qx, qy, qz, qw = quaternion  # scipy returns [x,y,z,w]
            
            f.write(f"{timestamp},{x_rot:.10f},{y_rot:.10f},{z_rot:.10f},{qw:.10f},{qx:.10f},{qy:.10f},{qz:.10f}\n")
    
    # Write dummy thrust data (required by Blackbird format)
    thrust_output = os.path.join(output_dir, 'thrust_data.csv')
    with open(thrust_output, 'w') as f:
        for i in range(len(times)):
            timestamp = int(times[i] * 1e6)
            f.write(f"{timestamp},0.0\n")
    
    print(f"  Converted {len(imu_data)} IMU samples and {len(poses)} poses")
    print(f"  Output: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert KITTI dataset to Blackbird format')
    parser.add_argument('--kitti_root', type=str, 
                        default='E:/Projects/Working/LLM/VO/VIFT-main/data/kitti_data',
                        help='Path to KITTI dataset root')
    parser.add_argument('--output_root', type=str,
                        default='./data/KITTI_Blackbird',
                        help='Output directory for converted data')
    parser.add_argument('--sequences', type=str, nargs='+',
                        default=['00', '01', '02', '04', '05', '06', '07', '08', '09', '10'],
                        help='Sequences to convert')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    
    print(f"Converting KITTI dataset from {args.kitti_root}")
    print(f"Output directory: {args.output_root}")
    print(f"Sequences: {args.sequences}\n")
    
    for seq in args.sequences:
        try:
            convert_kitti_sequence(args.kitti_root, seq, args.output_root)
        except Exception as e:
            print(f"  Error converting sequence {seq}: {e}")
    
    print("\nConversion complete!")
    print(f"\nTo use with Air-IO, update your config file:")
    print(f"  data_root: {args.output_root}")
    print(f"  data_drive: [seq_00, seq_01, ...]")

if __name__ == '__main__':
    main()
