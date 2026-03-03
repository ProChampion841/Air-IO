"""
Test IMU CSV file with trained AirIO model
"""
import os
import sys
import torch
import numpy as np
import pypose as pp
import argparse
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from model import get_model
from EKF.ekf import IMUEKF
from EKF.IMUstate import IMUstate


class CSVIMUReader:
    """Read IMU data from CSV file"""
    def __init__(self, csv_file):
        """
        Expected CSV format:
        # timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
        OR
        timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
        """
        self.csv_file = csv_file
        self.data = None
        self.current_idx = 0
        self.load_data()
    
    def load_data(self):
        """Load CSV file"""
        print(f"Loading IMU data from {self.csv_file}...")
        
        # Load CSV (skip header if starts with #)
        data = np.loadtxt(self.csv_file, delimiter=',', comments='#')
        
        if data.shape[1] == 7:
            # Format: timestamp, gx, gy, gz, ax, ay, az
            self.timestamps = data[:, 0]
            self.gyro = data[:, 1:4]  # rad/s
            self.acc = data[:, 4:7]   # m/s^2
        elif data.shape[1] == 6:
            # Format: gx, gy, gz, ax, ay, az (no timestamp)
            self.timestamps = np.arange(len(data)) * 0.005  # Assume 200Hz
            self.gyro = data[:, 0:3]
            self.acc = data[:, 3:6]
        else:
            raise ValueError(f"Unexpected CSV format: {data.shape[1]} columns")
        
        # Calculate dt
        self.dt = np.diff(self.timestamps, prepend=self.timestamps[0])
        
        print(f"✓ Loaded {len(self.timestamps)} samples")
        print(f"  Duration: {self.timestamps[-1] - self.timestamps[0]:.2f}s")
        print(f"  Average rate: {len(self.timestamps)/(self.timestamps[-1] - self.timestamps[0]):.1f} Hz")
    
    def read(self):
        """Read next sample"""
        if self.current_idx >= len(self.timestamps):
            return None
        
        data = {
            'acc': self.acc[self.current_idx],
            'gyro': self.gyro[self.current_idx],
            'dt': self.dt[self.current_idx],
            'timestamp': self.timestamps[self.current_idx]
        }
        self.current_idx += 1
        return data
    
    def reset(self):
        """Reset to beginning"""
        self.current_idx = 0


class SingleIMU(IMUstate):
    """IMU state model for EKF"""
    def __init__(self):
        super().__init__()

    def state_transition(self, state, input, dt, t=None):
        init_rot = pp.so3(state[..., :3]).Exp()
        bg = input[..., 6:9]
        ba = input[..., 9:12]

        w = (input[..., 0:3] - bg)
        a = (input[..., 3:6] - init_rot.Inv() @ self.gravity.double() - ba)
        Dr = pp.so3(w * dt).Exp()
        Dv = Dr @ a * dt
        Dp = Dv * dt + Dr @ a * 0.5 * dt**2
        R = (init_rot @ Dr).Log()
        V = state[..., 3:6] + init_rot @ Dv
        P = state[..., 6:9] + state[..., 3:6] * dt + init_rot @ Dp

        return torch.cat([R, V, P, bg, ba], dim=-1).tensor()

    def observation(self, state, input, dt, t=None):
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        velo = rot.Inv() @ nstate[..., 3:6]
        return velo


def test_imu_csv(csv_file, model_path, window_size=200, save_results=True):
    """Test IMU CSV file with trained model"""
    
    print("="*70)
    print("Testing IMU CSV with Trained Model")
    print("="*70)
    
    # Load IMU data
    print("\n[1/5] Loading IMU data...")
    imu_reader = CSVIMUReader(csv_file)
    
    # Load model
    print("\n[2/5] Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine network type from checkpoint or use default
    network_type = checkpoint.get('network_type', 'codewithrot')
    model = get_model(network_type)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Loaded {network_type} model from {model_path}")
    
    # Initialize EKF
    print("\n[3/5] Initializing EKF...")
    ekf_model = SingleIMU().double()
    Q = torch.eye(12, dtype=torch.float64) * 1e-4
    R = torch.eye(3, dtype=torch.float64) * 1e-2
    ekf = IMUEKF(ekf_model, Q=Q, R=R).double()
    
    state = torch.zeros(15, dtype=torch.float64)
    P = torch.eye(15, dtype=torch.float64) * 0.1
    
    # Initialize state (stationary at origin)
    init_rot = pp.identity_SO3(1)
    state[:3] = init_rot.Log()
    state[3:6] = torch.zeros(3)  # velocity
    state[6:9] = torch.zeros(3)  # position
    
    # Buffers
    imu_buffer = deque(maxlen=window_size)
    
    # Results storage
    positions = []
    velocities = []
    orientations = []
    timestamps = []
    
    # Process data
    print("\n[4/5] Processing IMU data...")
    pbar = tqdm(total=len(imu_reader.timestamps))
    
    while True:
        imu_data = imu_reader.read()
        if imu_data is None:
            break
        
        # Add to buffer
        imu_buffer.append(imu_data)
        
        # Run network when buffer is full
        velocity_obs = None
        if len(imu_buffer) >= window_size:
            # Prepare batch
            acc_seq = torch.stack([torch.tensor(d['acc'], dtype=torch.float32) for d in imu_buffer])
            gyro_seq = torch.stack([torch.tensor(d['gyro'], dtype=torch.float32) for d in imu_buffer])
            
            data = {
                "acc": acc_seq.unsqueeze(0).to(device),
                "gyro": gyro_seq.unsqueeze(0).to(device)
            }
            
            # Get rotation from current state for network input
            rot = pp.so3(state[:3]).Exp()
            rot_seq = rot.Log().repeat(window_size, 1).unsqueeze(0).to(device).float()
            
            # Run model
            with torch.no_grad():
                output = model(data, rot_seq)
                velocity = output['net_vel'][0, -1, :].cpu().double()
                velocity_obs = velocity
        
        # EKF update
        acc_t = torch.tensor(imu_data['acc'], dtype=torch.float64)
        gyro_t = torch.tensor(imu_data['gyro'], dtype=torch.float64)
        dt_t = torch.tensor(imu_data['dt'], dtype=torch.float64)
        
        d_bias_gyro = state[9:12]
        d_bias_acc = state[12:15]
        input_ekf = torch.cat([gyro_t, acc_t, d_bias_gyro, d_bias_acc], dim=-1)
        
        if velocity_obs is not None:
            state, P = ekf(state=state, obs=velocity_obs, input=input_ekf, P=P, dt=dt_t)
        else:
            state, P = ekf.state_propogate(state=state, input=input_ekf, P=P, dt=dt_t)
        
        # Save results
        rot = pp.so3(state[:3]).Exp()
        positions.append(state[6:9].clone().numpy())
        velocities.append(state[3:6].clone().numpy())
        orientations.append(rot.matrix().numpy())
        timestamps.append(imu_data['timestamp'])
        
        pbar.update(1)
    
    pbar.close()
    
    # Convert to arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    timestamps = np.array(timestamps)
    
    # Print statistics
    print("\n[5/5] Results:")
    print("-"*70)
    print(f"Total samples: {len(timestamps)}")
    print(f"Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"\nFinal state:")
    print(f"  Position: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}] m")
    print(f"  Velocity: [{velocities[-1, 0]:.3f}, {velocities[-1, 1]:.3f}, {velocities[-1, 2]:.3f}] m/s")
    print(f"  Total distance: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} m")
    
    # Visualize
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2D trajectory
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Position vs time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(timestamps - timestamps[0], positions[:, 0], label='X')
    ax3.plot(timestamps - timestamps[0], positions[:, 1], label='Y')
    ax3.plot(timestamps - timestamps[0], positions[:, 2], label='Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # Velocity vs time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(timestamps - timestamps[0], velocities[:, 0], label='Vx')
    ax4.plot(timestamps - timestamps[0], velocities[:, 1], label='Vy')
    ax4.plot(timestamps - timestamps[0], velocities[:, 2], label='Vz')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity vs Time')
    ax4.legend()
    ax4.grid(True)
    
    # Speed
    ax5 = fig.add_subplot(2, 3, 5)
    speed = np.linalg.norm(velocities, axis=1)
    ax5.plot(timestamps - timestamps[0], speed, 'b-')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Speed (m/s)')
    ax5.set_title('Speed vs Time')
    ax5.grid(True)
    
    # Distance
    ax6 = fig.add_subplot(2, 3, 6)
    distances = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    distances = np.insert(distances, 0, 0)
    ax6.plot(timestamps - timestamps[0], distances, 'b-')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance (m)')
    ax6.set_title(f'Total Distance: {distances[-1]:.2f} m')
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save
    if save_results:
        output_dir = 'test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        plot_file = os.path.join(output_dir, 'trajectory_plot.png')
        plt.savefig(plot_file, dpi=150)
        print(f"✓ Plot saved to {plot_file}")
        
        # Save numerical results
        results = {
            'timestamps': timestamps,
            'positions': positions,
            'velocities': velocities,
            'orientations': orientations
        }
        result_file = os.path.join(output_dir, 'odometry_results.npy')
        np.save(result_file, results)
        print(f"✓ Results saved to {result_file}")
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test IMU CSV file with trained model')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to IMU CSV file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--window-size', type=int, default=200,
                        help='Window size for network (default: 200)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    test_imu_csv(
        csv_file=args.csv,
        model_path=args.model,
        window_size=args.window_size,
        save_results=not args.no_save
    )
