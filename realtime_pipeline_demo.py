"""
Complete Real-Time IMU Odometry Pipeline Demo
Demonstrates the full AirIO pipeline with mock sensor data
"""
import os
import sys
import torch
import numpy as np
import pypose as pp
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from EKF.ekf import IMUEKF
from EKF.IMUstate import IMUstate


class MockIMUSensor:
    """Simulates IMU sensor with realistic motion"""
    def __init__(self, dt=0.005, duration=10.0):
        self.dt = dt
        self.duration = duration
        self.num_samples = int(duration / dt)
        self.current_idx = 0
        
        # Generate circular motion trajectory
        self.generate_trajectory()
        
    def generate_trajectory(self):
        """Generate ground truth trajectory and IMU measurements"""
        t = np.linspace(0, self.duration, self.num_samples)
        
        # Circular motion: radius=2m, period=5s
        omega = 2 * np.pi / 5.0  # angular velocity
        radius = 2.0
        
        # Ground truth position
        self.gt_pos = np.zeros((self.num_samples, 3))
        self.gt_pos[:, 0] = radius * np.cos(omega * t)
        self.gt_pos[:, 1] = radius * np.sin(omega * t)
        self.gt_pos[:, 2] = 0.1 * np.sin(2 * omega * t)  # slight vertical motion
        
        # Ground truth velocity
        self.gt_vel = np.zeros((self.num_samples, 3))
        self.gt_vel[:, 0] = -radius * omega * np.sin(omega * t)
        self.gt_vel[:, 1] = radius * omega * np.cos(omega * t)
        self.gt_vel[:, 2] = 0.1 * 2 * omega * np.cos(2 * omega * t)
        
        # Ground truth acceleration (body frame)
        self.gt_acc = np.zeros((self.num_samples, 3))
        self.gt_acc[:, 0] = -radius * omega**2 * np.cos(omega * t)
        self.gt_acc[:, 1] = -radius * omega**2 * np.sin(omega * t)
        self.gt_acc[:, 2] = -0.1 * 4 * omega**2 * np.sin(2 * omega * t)
        
        # Ground truth orientation (rotating around Z-axis)
        self.gt_rot = []
        for i in range(self.num_samples):
            yaw = omega * t[i]
            # Convert yaw to quaternion [w, x, y, z]
            qw = np.cos(yaw / 2)
            qx = 0.0
            qy = 0.0
            qz = np.sin(yaw / 2)
            quat = torch.tensor([qw, qx, qy, qz], dtype=torch.float64)
            self.gt_rot.append(pp.SO3(quat))
        
        # Simulate IMU measurements (with noise and bias)
        gyro_bias = np.array([0.01, -0.01, 0.005])
        acc_bias = np.array([0.05, -0.03, 0.02])
        
        self.imu_gyro = np.zeros((self.num_samples, 3))
        self.imu_acc = np.zeros((self.num_samples, 3))
        
        gravity = np.array([0, 0, 9.81007])
        
        for i in range(self.num_samples):
            # Gyroscope: angular velocity + bias + noise
            self.imu_gyro[i] = np.array([0, 0, omega]) + gyro_bias + np.random.randn(3) * 0.001
            
            # Accelerometer: specific force (acc - gravity) + bias + noise
            rot_inv = self.gt_rot[i].Inv().matrix().numpy()
            self.imu_acc[i] = self.gt_acc[i] + rot_inv @ gravity + acc_bias + np.random.randn(3) * 0.01
    
    def read(self):
        """Read next IMU sample"""
        if self.current_idx >= self.num_samples:
            return None
        
        data = {
            'acc': self.imu_acc[self.current_idx],
            'gyro': self.imu_gyro[self.current_idx],
            'dt': self.dt,
            'gt_pos': self.gt_pos[self.current_idx],
            'gt_vel': self.gt_vel[self.current_idx],
            'gt_rot': self.gt_rot[self.current_idx],
            'timestamp': self.current_idx * self.dt
        }
        self.current_idx += 1
        return data
    
    def reset(self):
        """Reset sensor to beginning"""
        self.current_idx = 0


class MockAirIMUNetwork:
    """Mock AirIMU network - simulates IMU correction (for demo)"""
    def __init__(self):
        self.window_size = 200
        
    def __call__(self, acc_seq, gyro_seq):
        """Simulate AirIMU correction"""
        corrected_acc = acc_seq * 0.98
        corrected_gyro = gyro_seq * 0.99
        
        batch_size, seq_len, _ = gyro_seq.shape
        rotations = []
        
        for b in range(batch_size):
            rot = pp.identity_SO3(1)
            for t in range(seq_len):
                omega = gyro_seq[b, t] * 0.005
                delta_rot = pp.so3(omega).Exp()
                rot = rot @ delta_rot
                rotations.append(rot)
        
        rotations = torch.stack([r.tensor() for r in rotations]).reshape(batch_size, seq_len, 4)
        
        return {
            'corrected_acc': corrected_acc,
            'corrected_gyro': corrected_gyro,
            'rotation': rotations,
            'acc_cov': torch.ones(batch_size, seq_len, 3) * 0.01,
            'gyro_cov': torch.ones(batch_size, seq_len, 3) * 0.001
        }


class MockAirIONetwork:
    """Mock AirIO network - simulates velocity prediction (for demo)"""
    def __init__(self):
        pass
        
    def __call__(self, corrected_acc, corrected_gyro, orientation):
        """Simulate velocity prediction"""
        batch_size, seq_len, _ = corrected_acc.shape
        velocity = torch.zeros(batch_size, seq_len, 3)
        
        for b in range(batch_size):
            vel = torch.zeros(3)
            for t in range(seq_len):
                vel += corrected_acc[b, t] * 0.005
                velocity[b, t] = vel
        
        return {
            'velocity': velocity,
            'cov': torch.ones(batch_size, seq_len, 3, 3) * 0.001
        }


class AirIMUNetwork:
    """Real AirIMU network - loads pre-trained model"""
    def __init__(self, model_path):
        from model import get_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model (adjust architecture as needed)
        self.model = get_model('airimu_network')  # Replace with actual AirIMU model name
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def __call__(self, acc_seq, gyro_seq):
        """Run AirIMU inference"""
        with torch.no_grad():
            acc_seq = acc_seq.to(self.device)
            gyro_seq = gyro_seq.to(self.device)
            
            # Run model
            output = self.model(acc_seq, gyro_seq)
            
            return {
                'corrected_acc': output['corrected_acc'].cpu(),
                'corrected_gyro': output['corrected_gyro'].cpu(),
                'rotation': output['rotation'].cpu(),
                'acc_cov': output.get('acc_cov', torch.ones_like(acc_seq) * 0.01).cpu(),
                'gyro_cov': output.get('gyro_cov', torch.ones_like(gyro_seq) * 0.001).cpu()
            }


class AirIONetwork:
    """Real AirIO network - loads pre-trained model"""
    def __init__(self, model_path, network_type='codewithrot'):
        from model import get_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        self.model = get_model(network_type)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def __call__(self, acc_seq, gyro_seq, orientation):
        """Run AirIO inference"""
        with torch.no_grad():
            acc_seq = acc_seq.to(self.device)
            gyro_seq = gyro_seq.to(self.device)
            orientation = orientation.to(self.device)
            
            # Run model
            output = self.model(acc_seq, gyro_seq, orientation)
            
            return {
                'velocity': output['velocity'].cpu(),
                'cov': output.get('cov', torch.eye(3).unsqueeze(0).unsqueeze(0) * 0.001).cpu()
            }


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


class RealtimeOdometryPipeline:
    """Complete real-time odometry pipeline"""
    def __init__(self, window_size=200, airimu_model_path=None, airio_model_path=None, use_mock=True):
        self.window_size = window_size
        self.imu_buffer = deque(maxlen=window_size)
        
        # Initialize networks (mock or real)
        if use_mock or airimu_model_path is None:
            print("Using MOCK networks (for demo only)")
            self.airimu_net = MockAirIMUNetwork()
            self.airio_net = MockAirIONetwork()
        else:
            print(f"Loading pre-trained models...")
            print(f"  AirIMU: {airimu_model_path}")
            print(f"  AirIO: {airio_model_path}")
            self.airimu_net = AirIMUNetwork(airimu_model_path)
            self.airio_net = AirIONetwork(airio_model_path)
        
        # Initialize EKF
        model = SingleIMU().double()
        Q = torch.eye(12, dtype=torch.float64) * 1e-4
        R = torch.eye(3, dtype=torch.float64) * 1e-2
        self.ekf = IMUEKF(model, Q=Q, R=R).double()
        
        self.state = torch.zeros(15, dtype=torch.float64)
        self.P = torch.eye(15, dtype=torch.float64) * 0.1
        
        # History
        self.est_pos_history = []
        self.est_vel_history = []
        self.est_rot_history = []
        
    def initialize(self, init_pos, init_rot, init_vel):
        """Initialize state"""
        self.state[:3] = init_rot.Log()
        self.state[3:6] = torch.tensor(init_vel, dtype=torch.float64)
        self.state[6:9] = torch.tensor(init_pos, dtype=torch.float64)
        
    def process_imu(self, imu_data):
        """Process single IMU measurement"""
        # Add to buffer
        self.imu_buffer.append(imu_data)
        
        # Convert to tensors
        acc_t = torch.tensor(imu_data['acc'], dtype=torch.float64)
        gyro_t = torch.tensor(imu_data['gyro'], dtype=torch.float64)
        dt_t = torch.tensor(imu_data['dt'], dtype=torch.float64)
        
        # Run networks if buffer is full
        velocity_obs = None
        if len(self.imu_buffer) >= self.window_size:
            velocity_obs = self._run_networks()
        
        # EKF update
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input_ekf = torch.cat([gyro_t, acc_t, d_bias_gyro, d_bias_acc], dim=-1)
        
        if velocity_obs is not None:
            # Propagate + Update
            self.state, self.P = self.ekf(
                state=self.state, 
                obs=velocity_obs, 
                input=input_ekf,
                P=self.P, 
                dt=dt_t
            )
        else:
            # Propagate only
            self.state, self.P = self.ekf.state_propogate(
                state=self.state,
                input=input_ekf,
                P=self.P,
                dt=dt_t
            )
        
        # Save history
        rot = pp.so3(self.state[:3]).Exp()
        self.est_pos_history.append(self.state[6:9].clone())
        self.est_vel_history.append(self.state[3:6].clone())
        self.est_rot_history.append(rot)
        
        return {
            'position': self.state[6:9].numpy(),
            'velocity': self.state[3:6].numpy(),
            'orientation': rot
        }
    
    def _run_networks(self):
        """Run AirIMU and AirIO networks"""
        # Prepare sequences
        acc_seq = torch.stack([torch.tensor(d['acc'], dtype=torch.float32) for d in self.imu_buffer])
        gyro_seq = torch.stack([torch.tensor(d['gyro'], dtype=torch.float32) for d in self.imu_buffer])
        
        acc_seq = acc_seq.unsqueeze(0)
        gyro_seq = gyro_seq.unsqueeze(0)
        
        # AirIMU
        airimu_out = self.airimu_net(acc_seq, gyro_seq)
        
        # AirIO
        airio_out = self.airio_net(
            airimu_out['corrected_acc'],
            airimu_out['corrected_gyro'],
            airimu_out['rotation']
        )
        
        # Return latest velocity (body frame)
        return airio_out['velocity'][0, -1, :].double()


def run_realtime_demo(airimu_model_path=None, airio_model_path=None, use_mock=True):
    """Run complete real-time pipeline demonstration
    
    Args:
        airimu_model_path: Path to pre-trained AirIMU model checkpoint
        airio_model_path: Path to pre-trained AirIO model checkpoint
        use_mock: If True, use mock networks instead of real models
    """
    print("="*70)
    print("Real-Time IMU Odometry Pipeline Demo")
    print("="*70)
    
    # Create mock sensor
    print("\n[1/5] Creating mock IMU sensor...")
    sensor = MockIMUSensor(dt=0.005, duration=10.0)
    print(f"  ✓ Generated {sensor.num_samples} samples (10 seconds @ 200Hz)")
    
    # Create pipeline
    print("\n[2/5] Initializing odometry pipeline...")
    pipeline = RealtimeOdometryPipeline(
        window_size=200,
        airimu_model_path=airimu_model_path,
        airio_model_path=airio_model_path,
        use_mock=use_mock
    )
    
    # Initialize with ground truth
    init_data = sensor.read()
    sensor.reset()
    pipeline.initialize(
        init_pos=init_data['gt_pos'],
        init_rot=init_data['gt_rot'],
        init_vel=init_data['gt_vel']
    )
    print("  ✓ Pipeline initialized")
    
    # Process all IMU data
    print("\n[3/5] Processing IMU data in real-time...")
    gt_positions = []
    errors = {'pos': [], 'vel': [], 'rot': []}
    
    pbar = tqdm(total=sensor.num_samples, desc="Processing")
    while True:
        imu_data = sensor.read()
        if imu_data is None:
            break
        
        # Process
        result = pipeline.process_imu(imu_data)
        
        # Calculate errors
        gt_positions.append(imu_data['gt_pos'])
        pos_error = np.linalg.norm(result['position'] - imu_data['gt_pos'])
        vel_error = np.linalg.norm(result['velocity'] - imu_data['gt_vel'])
        rot_error = (result['orientation'].Inv() @ imu_data['gt_rot']).Log().norm().item() * 180 / np.pi
        
        errors['pos'].append(pos_error)
        errors['vel'].append(vel_error)
        errors['rot'].append(rot_error)
        
        pbar.set_postfix({
            'Pos': f'{pos_error:.3f}m',
            'Vel': f'{vel_error:.3f}m/s',
            'Rot': f'{rot_error:.2f}°'
        })
        pbar.update(1)
    
    pbar.close()
    
    # Print statistics
    print("\n[4/5] Error Statistics:")
    print("-"*70)
    print(f"Position Error (m):")
    print(f"  Mean: {np.mean(errors['pos']):.4f}, Std: {np.std(errors['pos']):.4f}")
    print(f"  Min: {np.min(errors['pos']):.4f}, Max: {np.max(errors['pos']):.4f}")
    print(f"  Final: {errors['pos'][-1]:.4f}")
    print(f"\nVelocity Error (m/s):")
    print(f"  Mean: {np.mean(errors['vel']):.4f}, Std: {np.std(errors['vel']):.4f}")
    print(f"  Final: {errors['vel'][-1]:.4f}")
    print(f"\nRotation Error (degrees):")
    print(f"  Mean: {np.mean(errors['rot']):.4f}, Std: {np.std(errors['rot']):.4f}")
    print(f"  Final: {errors['rot'][-1]:.4f}")
    
    # Visualize results
    print("\n[5/5] Generating visualizations...")
    gt_positions = np.array(gt_positions)
    est_positions = torch.stack(pipeline.est_pos_history).numpy()
    
    # Create plots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(est_positions[:, 0], est_positions[:, 1], est_positions[:, 2], 'r--', label='Estimated', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2D trajectory (top view)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(gt_positions[:, 0], gt_positions[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax2.plot(est_positions[:, 0], est_positions[:, 1], 'r--', label='Estimated', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D Trajectory (Top View)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # Position errors
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(errors['pos'])
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Position Error (m)')
    ax3.set_title('Position Error Over Time')
    ax3.grid(True)
    
    # Velocity errors
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(errors['vel'])
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Velocity Error (m/s)')
    ax4.set_title('Velocity Error Over Time')
    ax4.grid(True)
    
    # Rotation errors
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(errors['rot'])
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Rotation Error (degrees)')
    ax5.set_title('Rotation Error Over Time')
    ax5.grid(True)
    
    # Error distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(errors['pos'], bins=50, alpha=0.7, label='Position')
    ax6.set_xlabel('Error (m)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Position Error Distribution')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    
    # Save
    output_dir = 'realtime_demo_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'realtime_pipeline_results.png'), dpi=150)
    print(f"  ✓ Results saved to {output_dir}/")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time IMU Odometry Pipeline Demo')
    parser.add_argument('--airimu-model', type=str, default=None,
                        help='Path to pre-trained AirIMU model checkpoint')
    parser.add_argument('--airio-model', type=str, default=None,
                        help='Path to pre-trained AirIO model checkpoint')
    parser.add_argument('--use-real-models', action='store_true',
                        help='Use real pre-trained models instead of mock networks')
    
    args = parser.parse_args()
    
    # Run demo
    run_realtime_demo(
        airimu_model_path=args.airimu_model,
        airio_model_path=args.airio_model,
        use_mock=not args.use_real_models
    )
