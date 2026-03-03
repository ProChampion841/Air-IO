import numpy as np
from scipy.spatial.transform import Rotation

def gps_orientation_to_quaternion(orientation, format='euler_deg'):
    """
    Convert GPS orientation to quaternion
    
    Args:
        orientation: GPS orientation data
        format: 'euler_deg' (roll,pitch,yaw in degrees)
                'euler_rad' (roll,pitch,yaw in radians)
                'heading_deg' (yaw only in degrees, roll=pitch=0)
                'heading_rad' (yaw only in radians, roll=pitch=0)
    
    Returns:
        qx, qy, qz, qw: Quaternion components
    """
    if format == 'euler_deg':
        # orientation should be Nx3 array: [roll, pitch, yaw] in degrees
        r = Rotation.from_euler('xyz', orientation, degrees=True)
    
    elif format == 'euler_rad':
        # orientation should be Nx3 array: [roll, pitch, yaw] in radians
        r = Rotation.from_euler('xyz', orientation, degrees=False)
    
    elif format == 'heading_deg':
        # orientation is 1D array: yaw/heading in degrees
        angles = np.column_stack([np.zeros(len(orientation)), 
                                  np.zeros(len(orientation)), 
                                  orientation])
        r = Rotation.from_euler('xyz', angles, degrees=True)
    
    elif format == 'heading_rad':
        # orientation is 1D array: yaw/heading in radians
        angles = np.column_stack([np.zeros(len(orientation)), 
                                  np.zeros(len(orientation)), 
                                  orientation])
        r = Rotation.from_euler('xyz', angles, degrees=False)
    
    quat = r.as_quat()  # Returns [qx, qy, qz, qw]
    return quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]


# Example usage
if __name__ == "__main__":
    # Example 1: Full orientation (roll, pitch, yaw) in degrees
    orientation = np.array([
        [0, 0, 45],    # roll=0, pitch=0, yaw=45°
        [0, 0, 90],
        [0, 0, 135]
    ])
    qx, qy, qz, qw = gps_orientation_to_quaternion(orientation, format='euler_deg')
    print("Full orientation (degrees):")
    print(f"qx={qx[0]:.4f}, qy={qy[0]:.4f}, qz={qz[0]:.4f}, qw={qw[0]:.4f}\n")
    
    # Example 2: Heading only in degrees
    heading = np.array([45, 90, 135])
    qx, qy, qz, qw = gps_orientation_to_quaternion(heading, format='heading_deg')
    print("Heading only (degrees):")
    print(f"qx={qx[0]:.4f}, qy={qy[0]:.4f}, qz={qz[0]:.4f}, qw={qw[0]:.4f}")
