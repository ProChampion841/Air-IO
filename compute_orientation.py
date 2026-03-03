import numpy as np
from scipy.spatial.transform import Rotation

def heading_from_gps(lat, lon):
    """
    Compute heading (yaw) from GPS trajectory
    Assumes pitch=0, roll=0 (flat movement)
    """
    # Convert to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Compute bearing between consecutive points
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    
    # Bearing formula
    y = np.sin(dlon) * np.cos(lat_rad[1:])
    x = np.cos(lat_rad[:-1]) * np.sin(lat_rad[1:]) - \
        np.sin(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.cos(dlon)
    heading = np.arctan2(y, x)
    
    # Pad first value (no previous point)
    heading = np.concatenate([[heading[0]], heading])
    
    return heading

def heading_to_quaternion(heading):
    """Convert heading (yaw) to quaternion (roll=0, pitch=0)"""
    r = Rotation.from_euler('z', heading, degrees=False)
    quat = r.as_quat()  # Returns [qx, qy, qz, qw]
    return quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]


## Option 2: From IMU Integration (if you have IMU data)
def quaternion_from_imu(gyro_x, gyro_y, gyro_z, dt):
    """
    Integrate gyroscope to get orientation
    
    Args:
        gyro_x, gyro_y, gyro_z: Angular velocity (rad/s)
        dt: Time step (seconds)
    
    Returns:
        qx, qy, qz, qw arrays
    """
    n = len(gyro_x)
    quats = np.zeros((n, 4))
    quats[0] = [0, 0, 0, 1]  # Identity quaternion
    
    for i in range(1, n):
        # Angular velocity vector
        omega = np.array([gyro_x[i], gyro_y[i], gyro_z[i]])
        angle = np.linalg.norm(omega) * dt
        
        if angle > 1e-8:
            axis = omega / np.linalg.norm(omega)
            delta_q = Rotation.from_rotvec(axis * angle).as_quat()
            
            # Quaternion multiplication
            q_prev = Rotation.from_quat(quats[i-1])
            q_delta = Rotation.from_quat(delta_q)
            quats[i] = (q_prev * q_delta).as_quat()
        else:
            quats[i] = quats[i-1]
    
    return quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]


## Option 3: From Euler Angles (if you have roll, pitch, yaw)
def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion
    
    Args:
        roll, pitch, yaw: Angles in radians
    
    Returns:
        qx, qy, qz, qw
    """
    r = Rotation.from_euler('xyz', np.column_stack([roll, pitch, yaw]), degrees=False)
    quat = r.as_quat()
    return quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]


# Example usage
if __name__ == "__main__":
    # Method 1: From GPS heading
    lat = np.array([40.7128, 40.7129, 40.7130, 40.7131])
    lon = np.array([-74.0060, -74.0059, -74.0058, -74.0057])
    
    heading = heading_from_gps(lat, lon)
    qx, qy, qz, qw = heading_to_quaternion(heading)
    
    print("Method 1 - GPS Heading:")
    print(f"Heading: {np.degrees(heading)}")
    print(f"Quaternion: qx={qx[0]:.4f}, qy={qy[0]:.4f}, qz={qz[0]:.4f}, qw={qw[0]:.4f}")
