"""
AirIO Model Inference - Input/Output Explanation
"""
import torch
import numpy as np
import pypose as pp

# ============================================================================
# MODEL INPUT
# ============================================================================

# 1. IMU Data (Required)
window_size = 200  # Typical window size
acc = torch.randn(1, window_size, 3)   # Accelerometer [batch, time, 3]
gyro = torch.randn(1, window_size, 3)  # Gyroscope [batch, time, 3]

# 2. Orientation (Optional - depends on network type)
# Three modes:
# - Mode 1: Ground truth orientation (for testing)
# - Mode 2: AirIMU corrected orientation
# - Mode 3: Raw IMU integration orientation

# Example: Create rotation sequence
rot_sequence = torch.zeros(1, window_size, 3)  # SO3 log format [batch, time, 3]

# Input dictionary
model_input = {
    "acc": acc,      # Shape: [1, 200, 3] - body frame or global frame
    "gyro": gyro,    # Shape: [1, 200, 3] - body frame or global frame
}

# If using attitude encoding (codewithrot network):
model_input_with_rot = {
    "acc": acc,
    "gyro": gyro,
    "rot": rot_sequence  # Shape: [1, 200, 3]
}

# ============================================================================
# MODEL OUTPUT
# ============================================================================

# Output dictionary
model_output = {
    "net_vel": predicted_velocity  # Shape: [1, 200, 3]
}

# The velocity is in the same frame as input:
# - If input is body-frame IMU → output is body-frame velocity
# - If input is global-frame IMU → output is global-frame velocity

# ============================================================================
# COORDINATE FRAMES
# ============================================================================

# Body Frame (body_coord):
# - acc: acceleration in body frame (m/s²)
# - gyro: angular velocity in body frame (rad/s)
# - output velocity: velocity in body frame (m/s)

# Global Frame (glob_coord):
# - acc: acceleration in global frame (m/s²) - gravity removed
# - gyro: angular velocity in body frame (rad/s)
# - output velocity: velocity in global frame (m/s)

# ============================================================================
# PRACTICAL EXAMPLE
# ============================================================================

def inference_example():
    """Complete inference example"""
    
    # Load your IMU data
    # Format: timestamps, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    imu_data = np.loadtxt('imu_data.csv', delimiter=',', comments='#')
    
    timestamps = imu_data[:, 0]
    acc_data = imu_data[:, 1:4]    # m/s²
    gyro_data = imu_data[:, 4:7]   # rad/s
    
    # Prepare sliding window
    window_size = 200
    predictions = []
    
    for i in range(len(imu_data) - window_size):
        # Extract window
        acc_window = torch.tensor(acc_data[i:i+window_size], dtype=torch.float32)
        gyro_window = torch.tensor(gyro_data[i:i+window_size], dtype=torch.float32)
        
        # Prepare input
        model_input = {
            "acc": acc_window.unsqueeze(0),   # Add batch dimension
            "gyro": gyro_window.unsqueeze(0)
        }
        
        # Run inference
        with torch.no_grad():
            output = model(model_input)
            velocity = output['net_vel'][0, -1, :]  # Get last timestep
        
        predictions.append(velocity.numpy())
    
    return np.array(predictions)


# ============================================================================
# SUMMARY
# ============================================================================

"""
INPUT:
------
- acc: [batch, window_size, 3] - Accelerometer data (m/s²)
- gyro: [batch, window_size, 3] - Gyroscope data (rad/s)
- rot (optional): [batch, window_size, 3] - Orientation in SO3 log format

OUTPUT:
-------
- net_vel: [batch, window_size, 3] - Predicted velocity (m/s)

NOTES:
------
1. Window size is typically 200-1000 frames
2. Use sliding window for continuous inference
3. Velocity is in same frame as input (body or global)
4. For position: integrate velocity with EKF (see EKF/IMUofflinerunner.py)
"""

print(__doc__)
