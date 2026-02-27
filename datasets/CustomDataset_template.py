"""
Template Dataset Class for Custom Data
Copy this file and modify for your dataset format
"""
import os
import torch
import numpy as np
import pypose as pp
from scipy.interpolate import interp1d
from datasets.EuRoCdataset import Euroc


class CustomDataset(Euroc):
    """
    Template for custom dataset
    
    Required files in each sequence folder:
    - imu_data.csv: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
    - groundTruthPoses.csv: timestamp, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z
    - thrust_data.csv (optional): timestamp, throttle
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self, data_path):
        """Load and process your dataset
        
        Args:
            data_path: Path to sequence folder (e.g., data/MyDataset/train/sequence1)
        """
        print(f"Loading data from {data_path}")
        
        # ============================================================
        # STEP 1: Load IMU data
        # ============================================================
        imu_file = os.path.join(data_path, 'imu_data.csv')
        
        # Load CSV (skip header if present)
        imu_data = np.loadtxt(imu_file, delimiter=',', skiprows=1)
        
        # Parse columns: timestamp, gx, gy, gz, ax, ay, az
        imu_timestamps = imu_data[:, 0]
        gyro_raw = imu_data[:, 1:4]  # rad/s
        acc_raw = imu_data[:, 4:7]   # m/s^2
        
        print(f"  Loaded {len(imu_timestamps)} IMU samples")
        
        # ============================================================
        # STEP 2: Load ground truth poses
        # ============================================================
        gt_file = os.path.join(data_path, 'groundTruthPoses.csv')
        
        gt_data = np.loadtxt(gt_file, delimiter=',', skiprows=1)
        
        # Parse columns: timestamp, px, py, pz, qw, qx, qy, qz
        gt_timestamps = gt_data[:, 0]
        gt_positions = gt_data[:, 1:4]  # meters
        gt_quaternions = gt_data[:, 4:8]  # [w, x, y, z]
        
        print(f"  Loaded {len(gt_timestamps)} ground truth samples")
        
        # ============================================================
        # STEP 3: Interpolate to common timestamps
        # ============================================================
        # Use IMU timestamps as reference
        timestamps = imu_timestamps
        
        # Interpolate ground truth to IMU timestamps
        gt_pos_interp = self._interpolate_data(gt_timestamps, gt_positions, timestamps)
        gt_quat_interp = self._interpolate_quaternions(gt_timestamps, gt_quaternions, timestamps)
        
        # ============================================================
        # STEP 4: Compute velocity from position
        # ============================================================
        velocity = self._compute_velocity(gt_pos_interp, timestamps)
        
        # ============================================================
        # STEP 5: Convert to PyTorch tensors
        # ============================================================
        dt = np.diff(timestamps, prepend=timestamps[0])
        
        self.data = {
            "dt": dt,
            "time": timestamps,
            "acc": torch.tensor(acc_raw, dtype=torch.float32),
            "gyro": torch.tensor(gyro_raw, dtype=torch.float32),
            "gt_orientation": pp.SO3(torch.tensor(gt_quat_interp, dtype=torch.float32)),
            "gt_translation": torch.tensor(gt_pos_interp, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32)
        }
        
        print(f"  Final dataset: {len(timestamps)} samples")
        
        # ============================================================
        # STEP 6: Apply transformations (required)
        # ============================================================
        # Set orientation (from AirIMU or ground truth)
        self.set_orientation(self.rot_path, data_path, self.rot_type)
        
        # Transform to body/global frame
        self.update_coordinate(self.coordinate, self.mode)
        
        # Remove gravity if needed
        self.remove_gravity(self.remove_g)
    
    def _interpolate_data(self, src_timestamps, src_data, dst_timestamps):
        """Interpolate data to new timestamps
        
        Args:
            src_timestamps: Original timestamps (N,)
            src_data: Original data (N, D)
            dst_timestamps: Target timestamps (M,)
            
        Returns:
            Interpolated data (M, D)
        """
        if src_data.ndim == 1:
            src_data = src_data[:, np.newaxis]
        
        interpolated = np.zeros((len(dst_timestamps), src_data.shape[1]))
        
        for i in range(src_data.shape[1]):
            f = interp1d(src_timestamps, src_data[:, i], 
                        kind='linear', fill_value='extrapolate')
            interpolated[:, i] = f(dst_timestamps)
        
        return interpolated
    
    def _interpolate_quaternions(self, src_timestamps, src_quats, dst_timestamps):
        """Interpolate quaternions using SLERP
        
        Args:
            src_timestamps: Original timestamps (N,)
            src_quats: Original quaternions (N, 4) [w, x, y, z]
            dst_timestamps: Target timestamps (M,)
            
        Returns:
            Interpolated quaternions (M, 4)
        """
        from scipy.spatial.transform import Rotation, Slerp
        
        # Create rotation objects
        rotations = Rotation.from_quat(src_quats[:, [1, 2, 3, 0]])  # scipy uses [x,y,z,w]
        
        # Create SLERP interpolator
        slerp = Slerp(src_timestamps, rotations)
        
        # Interpolate
        interp_rotations = slerp(dst_timestamps)
        interp_quats = interp_rotations.as_quat()  # [x, y, z, w]
        
        # Convert back to [w, x, y, z]
        return np.column_stack([interp_quats[:, 3], interp_quats[:, :3]])
    
    def _compute_velocity(self, positions, timestamps):
        """Compute velocity from position using finite differences
        
        Args:
            positions: Position array (N, 3)
            timestamps: Timestamp array (N,)
            
        Returns:
            Velocity array (N, 3)
        """
        velocity = np.zeros_like(positions)
        
        # Forward difference for first point
        velocity[0] = (positions[1] - positions[0]) / (timestamps[1] - timestamps[0])
        
        # Central difference for middle points
        for i in range(1, len(positions) - 1):
            dt = timestamps[i+1] - timestamps[i-1]
            velocity[i] = (positions[i+1] - positions[i-1]) / dt
        
        # Backward difference for last point
        velocity[-1] = (positions[-1] - positions[-2]) / (timestamps[-1] - timestamps[-2])
        
        # Optional: Apply smoothing
        from scipy.ndimage import gaussian_filter1d
        for i in range(3):
            velocity[:, i] = gaussian_filter1d(velocity[:, i], sigma=2.0)
        
        return velocity


# ============================================================
# Example usage and testing
# ============================================================
if __name__ == '__main__':
    """Test your dataset class"""
    
    # Test parameters
    test_config = {
        'data_root': 'data/MyDataset',
        'data_name': 'train/sequence1',
        'mode': 'train',
        'window_size': 1000,
        'step_size': 10,
        'coordinate': 'body_coord',
        'remove_g': False,
        'rot_type': None,
        'rot_path': None,
        'gravity': 9.81007
    }
    
    print("="*70)
    print("Testing Custom Dataset Class")
    print("="*70)
    
    try:
        # Create dataset instance
        dataset = CustomDataset(**test_config)
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Window size: {test_config['window_size']}")
        print(f"  Number of windows: {len(dataset)}")
        
        # Test getting one sample
        sample = dataset[0]
        
        print(f"\n✓ Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Test data ranges
        print(f"\n✓ Data ranges:")
        print(f"  Gyro: [{sample['gyro'].min():.3f}, {sample['gyro'].max():.3f}] rad/s")
        print(f"  Acc: [{sample['acc'].min():.3f}, {sample['acc'].max():.3f}] m/s²")
        print(f"  Velocity: [{sample['velocity'].min():.3f}, {sample['velocity'].max():.3f}] m/s")
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
