"""
Real-time IMU Odometry Runner
Processes live IMU sensor data for real-time odometry estimation
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import torch
import pypose as pp
import numpy as np
from collections import deque
from ekf import IMUEKF
from IMUstate import IMUstate

class SingleIMU(IMUstate):
    """IMU state model for EKF"""
    def __init__(self):
        super().__init__()

    def state_transition(self, state: torch.Tensor, input: torch.Tensor, dt: torch.Tensor, t: torch.Tensor=None):
        init_rot = pp.so3(state[..., :3]).Exp()
        bg = input[..., 6:9]
        ba = input[..., 9:12]

        w = (input[..., 0:3]-bg)
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

class RealtimeOdometry:
    """
    Real-time IMU-only odometry system
    
    Usage:
        odometry = RealtimeOdometry(airimu_model, airio_model, window_size=1000)
        odometry.initialize(init_pos, init_rot, init_vel)
        
        # In your sensor callback:
        result = odometry.process_imu(acc, gyro, dt)
        position = result['position']
        orientation = result['orientation']
    """
    
    def __init__(self, airimu_model, airio_model, window_size=1000, device='cpu'):
        """
        Args:
            airimu_model: Loaded AirIMU network model
            airio_model: Loaded AirIO network model
            window_size: Sequence length for network inference
            device: 'cpu' or 'cuda'
        """
        self.airimu_model = airimu_model.to(device).eval()
        self.airio_model = airio_model.to(device).eval()
        self.window_size = window_size
        self.device = device
        
        # Sliding window buffers
        self.imu_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # EKF setup
        model = SingleIMU().double()
        Q = torch.eye(12, dtype=torch.float64) * 1e-12
        R = torch.eye(3, dtype=torch.float64) * 1e-3
        self.ekf = IMUEKF(model, Q=Q, R=R).double()
        
        self.state = torch.zeros(15, dtype=torch.float64)
        self.P = torch.eye(15, dtype=torch.float64) * 0.01
        self.initialized = False
        self.current_time = 0.0
        
    def initialize(self, init_pos, init_rot, init_vel, gravity=9.81007):
        """
        Initialize odometry state
        
        Args:
            init_pos: [x, y, z] initial position
            init_rot: SO3 or quaternion initial rotation
            init_vel: [vx, vy, vz] initial velocity
            gravity: gravity magnitude
        """
        if isinstance(init_rot, pp.SO3):
            rot_log = init_rot.Log()
        else:
            rot_log = pp.SO3(init_rot).Log()
            
        self.state[:3] = rot_log
        self.state[3:6] = torch.tensor(init_vel, dtype=torch.float64)
        self.state[6:9] = torch.tensor(init_pos, dtype=torch.float64)
        self.state[9:] = 0.0  # Zero bias initialization
        
        self.gravity = torch.tensor([0., 0., gravity], dtype=torch.float64)
        self.initialized = True
        
    def process_imu(self, acc, gyro, dt):
        """
        Process single IMU measurement in real-time
        
        Args:
            acc: [ax, ay, az] accelerometer reading (m/s^2)
            gyro: [wx, wy, wz] gyroscope reading (rad/s)
            dt: time step (seconds)
            
        Returns:
            dict: {
                'position': [x, y, z],
                'orientation': SO3 rotation,
                'velocity': [vx, vy, vz],
                'bias_gyro': [bgx, bgy, bgz],
                'bias_acc': [bax, bay, baz],
                'covariance': 15x15 matrix
            }
        """
        if not self.initialized:
            raise RuntimeError("Call initialize() before processing IMU data")
        
        # Add to buffer
        self.imu_buffer.append({'acc': acc, 'gyro': gyro, 'dt': dt})
        self.current_time += dt
        self.time_buffer.append(self.current_time)
        
        # Convert to tensors
        acc_t = torch.tensor(acc, dtype=torch.float64)
        gyro_t = torch.tensor(gyro, dtype=torch.float64)
        dt_t = torch.tensor(dt, dtype=torch.float64)
        
        # Run networks if buffer is full
        velocity_obs = None
        if len(self.imu_buffer) >= self.window_size:
            velocity_obs = self._run_networks()
        
        # EKF update
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input_ekf = torch.cat([gyro_t, acc_t, d_bias_gyro, d_bias_acc], dim=-1)
        
        if velocity_obs is not None:
            # Propagate + Update with velocity measurement
            self.state, self.P = self.ekf(
                state=self.state, 
                obs=velocity_obs, 
                input=input_ekf,
                P=self.P, 
                dt=dt_t
            )
        else:
            # Propagate only (no measurement yet)
            self.state, self.P = self.ekf.state_propogate(
                state=self.state,
                input=input_ekf,
                P=self.P,
                dt=dt_t
            )
        
        # Extract results
        rot = pp.so3(self.state[:3]).Exp()
        return {
            'position': self.state[6:9].numpy(),
            'orientation': rot,
            'velocity': self.state[3:6].numpy(),
            'bias_gyro': self.state[9:12].numpy(),
            'bias_acc': self.state[12:15].numpy(),
            'covariance': self.P.numpy(),
            'timestamp': self.current_time
        }
    
    def _run_networks(self):
        """Run AirIMU and AirIO networks on buffered data"""
        # Prepare sequence data
        acc_seq = torch.stack([torch.tensor(d['acc']) for d in self.imu_buffer])
        gyro_seq = torch.stack([torch.tensor(d['gyro']) for d in self.imu_buffer])
        
        # Add batch dimension and move to device
        acc_seq = acc_seq.unsqueeze(0).to(self.device).float()
        gyro_seq = gyro_seq.unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            # AirIMU: correct IMU and get orientation
            airimu_out = self.airimu_model(acc_seq, gyro_seq)
            corrected_acc = airimu_out['corrected_acc']
            corrected_gyro = airimu_out['corrected_gyro']
            orientation = airimu_out['rotation']
            
            # AirIO: predict velocity
            airio_out = self.airio_model(corrected_acc, corrected_gyro, orientation)
            velocity = airio_out['velocity']
        
        # Return latest velocity prediction (body frame)
        return velocity[0, -1, :].cpu().double()
    
    def get_trajectory(self):
        """Get current position as trajectory point"""
        rot = pp.so3(self.state[:3]).Exp()
        return {
            'position': self.state[6:9].numpy(),
            'orientation': rot.matrix().numpy(),
            'timestamp': self.current_time
        }


# Example usage
if __name__ == '__main__':
    """
    Example: Real-time processing with live IMU sensor
    """
    
    # 1. Load pre-trained models
    print("Loading models...")
    # TODO: Load your trained AirIMU and AirIO models
    # airimu_model = torch.load('path/to/airimu_model.ckpt')
    # airio_model = torch.load('path/to/airio_model.ckpt')
    
    # 2. Initialize odometry system
    # odometry = RealtimeOdometry(airimu_model, airio_model, window_size=1000)
    # odometry.initialize(
    #     init_pos=[0, 0, 0],
    #     init_rot=pp.identity_SO3(1),
    #     init_vel=[0, 0, 0]
    # )
    
    # 3. Process IMU data in real-time
    # In your sensor callback or main loop:
    """
    def imu_callback(acc, gyro, dt):
        result = odometry.process_imu(acc, gyro, dt)
        
        print(f"Position: {result['position']}")
        print(f"Velocity: {result['velocity']}")
        print(f"Orientation: {result['orientation']}")
        
        # Use result for navigation, control, etc.
        return result
    """
    
    # 4. Simulate real-time processing
    print("Simulating real-time IMU processing...")
    print("Connect to your IMU sensor and call process_imu() in the callback")
    print("Example: result = odometry.process_imu(acc, gyro, dt)")
